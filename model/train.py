"""
Training script for the GATv2 Graph Autoencoder with combined
Reconstruction + Triplet Margin loss.

Loss function
-------------
  L = L_recon + λ * L_triplet

  L_recon  : GAE link-reconstruction loss (log-sigmoid on inner products)
  L_triplet: TripletMarginLoss over (anchor, positive, negative) triples
             sampled from FlavorGraph edges.
             Positive pairs  = connected by a FlavorGraph edge.
             Negative samples = random nodes (graph is sparse, so accidental
             false-negatives are statistically rare and acceptable).

Run from the repo root:
    python model/train.py
or from within model/:
    python train.py
"""

import sys
from pathlib import Path

import torch
from torch_geometric.nn import GAE

ROOT = Path(__file__).resolve().parent.parent

# Allow absolute imports from both model/ and the project root
sys.path.insert(0, str(Path(__file__).parent))   # for graph_autoencoder
sys.path.insert(0, str(ROOT))                     # for data.load_graph

from graph_autoencoder import GATv2Encoder        # noqa: E402
from data.load_graph import load_graph            # noqa: E402

# ─────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────
EMBEDDING_DIM   = 128
HEADS           = 4
EPOCHS          = 200
LR              = 0.005
TRIPLET_MARGIN  = 0.3
TRIPLET_WEIGHT  = 0.5     # λ — weight of triplet loss relative to recon loss
N_TRIPLETS      = 1024    # triplets sampled per mini-batch
SMILES_PATH     = ROOT / "data/ingredient_smiles.csv"

# ─────────────────────────────────────────────────────────────────
# 1. Load graph (Morgan fingerprints if SMILES available, else zeros)
# ─────────────────────────────────────────────────────────────────
data, nodes_df = load_graph(
    smiles_path=SMILES_PATH if SMILES_PATH.exists() else None
)

in_channels = data.x.shape[1]
n_nodes     = data.x.shape[0]

# ─────────────────────────────────────────────────────────────────
# 2. Triplet sampler
# ─────────────────────────────────────────────────────────────────
def sample_triplets(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    n: int = N_TRIPLETS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample (anchor, positive, negative) index triples.

    Positives come from FlavorGraph edges — ingredients connected by shared
    flavor compounds.  Negatives are random nodes; because the graph is sparse
    (~147 K edges over 8 K² ≈ 69 M possible pairs, density ≈ 0.2 %), the
    probability of sampling an accidental false-negative is negligible.

    Returns
    -------
    anchors, positives, negatives : LongTensor of shape [n]
    """
    num_edges = edge_index.shape[1]
    edge_idx  = torch.randint(0, num_edges, (n,), device=z.device)
    anchors   = edge_index[0, edge_idx]
    positives = edge_index[1, edge_idx]
    negatives = torch.randint(0, z.shape[0], (n,), device=z.device)
    return anchors, positives, negatives


# ─────────────────────────────────────────────────────────────────
# 3. Model, optimiser, loss
# ─────────────────────────────────────────────────────────────────
model          = GAE(GATv2Encoder(in_channels=in_channels, out_channels=EMBEDDING_DIM, heads=HEADS))
optimizer      = torch.optim.Adam(model.parameters(), lr=LR)
triplet_loss_fn = torch.nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)


# ─────────────────────────────────────────────────────────────────
# 4. Training loop
# ─────────────────────────────────────────────────────────────────
def train_step() -> tuple[float, float]:
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.x, data.edge_index)

    # Reconstruction loss — predicts which edges exist via inner products
    recon_loss = model.recon_loss(z, data.edge_index)

    # Triplet margin loss — pulls flavor-paired ingredients together in the
    # latent space and pushes random negatives apart
    a, p, n   = sample_triplets(z, data.edge_index)
    trip_loss = triplet_loss_fn(z[a], z[p], z[n])

    loss = recon_loss + TRIPLET_WEIGHT * trip_loss
    loss.backward()
    optimizer.step()

    return float(recon_loss), float(trip_loss)


print(f"\nTraining GATv2 encoder  (nodes={n_nodes}, feature_dim={in_channels}, emb_dim={EMBEDDING_DIM})")
print(f"Epochs={EPOCHS}  LR={LR}  triplet_λ={TRIPLET_WEIGHT}  margin={TRIPLET_MARGIN}\n")

for epoch in range(EPOCHS):
    recon_l, trip_l = train_step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Recon: {recon_l:.4f} | Triplet: {trip_l:.4f}")


# ─────────────────────────────────────────────────────────────────
# 5. Save artifacts
# ─────────────────────────────────────────────────────────────────
artifacts_dir = ROOT / "artifacts"
artifacts_dir.mkdir(exist_ok=True)

# Model weights
model_path = artifacts_dir / "molecular_mirror_weights.pth"
torch.save(model.state_dict(), model_path)
print(f"\nModel saved  → {model_path}")

# Pre-computed embeddings (128-dim) — loaded by agents/inference.py at startup
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.edge_index)

embeddings_path = artifacts_dir / "ingredient_embeddings.pt"
torch.save(z, embeddings_path)
print(f"Embeddings saved → {embeddings_path}  shape={tuple(z.shape)}")
