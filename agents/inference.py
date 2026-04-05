"""
Inference utilities for Molecular Mirror.

Loads the pre-computed 128-dim GATv2 embeddings at import time and exposes
`get_molecular_mirrors` for use both directly and as an AG2 agent tool.

The embeddings are dimension-agnostic: this file works with any checkpoint
produced by model/train.py regardless of the configured EMBEDDING_DIM.
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent

# Load once at module import — avoids reloading on every agent tool call.
z     = torch.load(ROOT / "artifacts/ingredient_embeddings.pt", weights_only=True)
nodes = pd.read_csv(ROOT / "data/nodes_191120.csv")

_EMBED_DIM = z.shape[1]  # 128 with the GATv2 encoder


def get_molecular_mirrors(
    target_name: Annotated[str, "name of the food ingredient to find mirrors for"],
    top_k: int = 5,
) -> list[tuple[str, float]] | str:
    """
    Find the ``top_k`` most molecularly similar ingredients to ``target_name``
    using cosine similarity in the GATv2 latent space.

    Parameters
    ----------
    target_name : str
        Full or partial ingredient name (case-insensitive substring match).
    top_k : int
        Number of nearest neighbours to return (default 5).

    Returns
    -------
    List of (ingredient_name, similarity_score) tuples, or an error string
    if the ingredient is not found.
    """
    # 1. Locate target node (case-insensitive substring match)
    matches = nodes[nodes["name"].str.contains(target_name, case=False, na=False)]
    if matches.empty:
        return f"Ingredient '{target_name}' not found in FlavorGraph."

    target_idx  = matches.index[0]
    actual_name = nodes.iloc[target_idx]["name"]

    # 2. Cosine similarity against all nodes in the {_EMBED_DIM}-dim space
    target_vec  = z[target_idx].unsqueeze(0)          # [1, embed_dim]
    similarities = F.cosine_similarity(target_vec, z)  # [N]

    # 3. Top-(k+1) to skip the target itself (index 0 after sorting)
    values, indices = torch.topk(similarities, k=top_k + 1)

    print(f"\n--- Molecular Mirrors for: {actual_name} (embed_dim={_EMBED_DIM}) ---")
    results: list[tuple[str, float]] = []
    for rank in range(1, len(indices)):   # skip rank 0 (the target itself)
        name  = nodes.iloc[indices[rank].item()]["name"]
        score = values[rank].item()
        results.append((name, score))
        print(f"  {rank}. {name}  (similarity: {score:.4f})")

    return results
