"""
Data pipeline for FlavorGraph with SMILES-based Morgan fingerprint featurization.

Node feature strategy (in priority order)
------------------------------------------
1. If ``smiles_path`` points to a CSV with columns [name, smiles] or
   [name, smiles_list], compute 2048-bit Morgan fingerprints (radius 2) via
   RDKit.  Ingredients with multiple compounds use the mean fingerprint.
2. If the SMILES file is missing or an ingredient has no entry, that node's
   feature vector is zeros (the model will learn a bias correction).

Generating ingredient_smiles.csv
---------------------------------
You can create this file by:
  a. Downloading the FlavorGraph compound–ingredient mapping from the official
     FlavorGraph dataset (compound nodes link ingredients to SMILES via the
     knowledge graph edges).
  b. Querying PubChem with ingredient names:
       from pubchempy import get_compounds
       compounds = get_compounds('vanilla', 'name')
       smiles = compounds[0].isomeric_smiles

Expected CSV schema (either variant is supported):
  Single compound per row  → columns: name, smiles
  Multiple compounds       → columns: name, smiles_list   (semicolon-separated)
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[WARNING] RDKit not found. Install with: pip install rdkit")

ROOT = Path(__file__).parent.parent

FINGERPRINT_DIM = 2048
MORGAN_RADIUS = 2


# ─────────────────────────────────────────────────────────────────
# Low-level fingerprint helpers
# ─────────────────────────────────────────────────────────────────

def smiles_to_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """
    Convert a SMILES string to a 2048-bit Morgan fingerprint.

    Returns None if the SMILES cannot be parsed.
    """
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=MORGAN_RADIUS, nBits=FINGERPRINT_DIM
    )
    return np.array(fp, dtype=np.float32)


def mean_fingerprint(smiles_list: list[str]) -> Optional[np.ndarray]:
    """
    Compute the mean Morgan fingerprint over a list of SMILES strings.

    Useful when an ingredient is characterised by multiple flavor compounds.
    Invalid SMILES are silently skipped.  Returns None if none are valid.
    """
    fps = [smiles_to_fingerprint(s) for s in smiles_list]
    fps = [f for f in fps if f is not None]
    if not fps:
        return None
    return np.mean(fps, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# Node feature construction
# ─────────────────────────────────────────────────────────────────

def build_node_features(
    nodes_df: pd.DataFrame,
    smiles_df: Optional[pd.DataFrame] = None,
) -> torch.Tensor:
    """
    Build a [N, FINGERPRINT_DIM] feature matrix from Morgan fingerprints.

    Parameters
    ----------
    nodes_df  : DataFrame with at least columns [node_id, name]
    smiles_df : Optional DataFrame with [name, smiles] or [name, smiles_list]

    Returns
    -------
    Tensor of shape [N, 2048], dtype=float32
    """
    n = len(nodes_df)
    features = np.zeros((n, FINGERPRINT_DIM), dtype=np.float32)

    if smiles_df is None or not RDKIT_AVAILABLE:
        reason = "no SMILES file provided" if smiles_df is None else "RDKit unavailable"
        print(
            f"[WARNING] Featurization fallback ({reason}). "
            f"Using zero-initialised {FINGERPRINT_DIM}-dim features. "
            "Provide data/ingredient_smiles.csv to enable Morgan fingerprints."
        )
        return torch.from_numpy(features)

    # Build name → list[smiles] mapping (case-insensitive)
    smiles_map: dict[str, list[str]] = {}
    if "smiles_list" in smiles_df.columns:
        for _, row in smiles_df.iterrows():
            key = str(row["name"]).lower()
            smiles_map[key] = str(row["smiles_list"]).split(";")
    elif "smiles" in smiles_df.columns:
        for _, row in smiles_df.iterrows():
            key = str(row["name"]).lower()
            smiles_map[key] = [str(row["smiles"])]
    else:
        print("[WARNING] smiles_df has neither 'smiles' nor 'smiles_list' column. Using zeros.")
        return torch.from_numpy(features)

    found = 0
    for i, row in nodes_df.iterrows():
        key = str(row["name"]).lower()
        # Also try with spaces replacing underscores (FlavorGraph uses underscores)
        key_spaced = key.replace("_", " ")
        smiles_list = smiles_map.get(key) or smiles_map.get(key_spaced)
        if smiles_list:
            fp = mean_fingerprint(smiles_list)
            if fp is not None:
                features[i] = fp
                found += 1

    missing = n - found
    print(
        f"[INFO] Morgan fingerprints: {found}/{n} nodes featurised "
        f"({missing} missing → zero vectors)."
    )
    return torch.from_numpy(features)


# ─────────────────────────────────────────────────────────────────
# Graph loader
# ─────────────────────────────────────────────────────────────────

def load_graph(
    smiles_path: Optional[Path] = None,
) -> Tuple[Data, pd.DataFrame]:
    """
    Load FlavorGraph and return a PyG Data object with Morgan fingerprint features.

    Parameters
    ----------
    smiles_path : Optional path to a CSV with ingredient SMILES.
                  Defaults to data/ingredient_smiles.csv if it exists.

    Returns
    -------
    data     : torch_geometric.data.Data  (x, edge_index)
    nodes_df : pd.DataFrame               (for downstream name look-ups)
    """
    nodes_df = pd.read_csv(ROOT / "data/nodes_191120.csv")
    edges_df = pd.read_csv(ROOT / "data/edges_191120.csv")

    # --- Edge index (0-indexed) ---
    node_map = {old_id: i for i, old_id in enumerate(nodes_df["node_id"])}
    valid_edges = [
        (node_map[s], node_map[t])
        for s, t in zip(edges_df["id_1"], edges_df["id_2"])
        if s in node_map and t in node_map
    ]
    edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()

    # --- Node features ---
    # Resolve SMILES file: explicit arg > default path
    if smiles_path is None:
        default_path = ROOT / "data/ingredient_smiles.csv"
        smiles_path = default_path if default_path.exists() else None

    smiles_df = None
    if smiles_path is not None and smiles_path.exists():
        smiles_df = pd.read_csv(smiles_path)
        print(f"[INFO] Loaded SMILES from {smiles_path} ({len(smiles_df)} rows)")

    x = build_node_features(nodes_df, smiles_df)

    data = Data(x=x, edge_index=edge_index)
    print(
        f"[INFO] Graph loaded: {len(nodes_df)} nodes, "
        f"{edge_index.shape[1]} edges, feature_dim={x.shape[1]}"
    )
    return data, nodes_df
