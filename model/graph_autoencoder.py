"""
Graph Autoencoder — GATv2 encoder with DeepChem-inspired projection head.

Architecture
------------
Input  : node features  [N, in_channels]
Conv1  : GATv2Conv(in_channels → 64/head × 4 heads, concat=True)  → [N, 256]
Conv2  : GATv2Conv(256 → 128, heads=1, concat=False)               → [N, 128]
Proj   : Linear(128→128) → BN → ReLU → Linear(128→128)             → [N, 128]

The projection head follows the DeepChem contrastive pre-training convention:
the backbone (conv1+conv2) produces general-purpose geometry while the head
absorbs task-specific metric structure during triplet training.
Cosine similarity at inference time operates on the projected embeddings.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

EMBEDDING_DIM = 128
HEADS = 4
_HIDDEN_PER_HEAD = 64  # → 64 * HEADS = 256 after concat in layer 1


class GATv2Encoder(torch.nn.Module):
    """
    GATv2 encoder with multi-head attention and a projection head.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features (2048 for Morgan fingerprints).
    out_channels : int
        Final embedding dimension. Default: 128.
    heads : int
        Number of attention heads in the first GAT layer. Default: 4.
    dropout : float
        Attention coefficient dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = EMBEDDING_DIM,
        heads: int = HEADS,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Layer 1: multi-head GATv2 — concat=True → _HIDDEN_PER_HEAD * heads output
        self.conv1 = GATv2Conv(
            in_channels,
            _HIDDEN_PER_HEAD,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        # Layer 2: single-head GATv2 — concat=False → out_channels output
        self.conv2 = GATv2Conv(
            _HIDDEN_PER_HEAD * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        # DeepChem-inspired projection head:
        # 2-layer MLP with BatchNorm1d keeps the backbone representations general
        # while the projection absorbs task-specific metric geometry (triplet loss).
        self.projection = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : [N, in_channels]   node feature matrix
        edge_index : [2, E]             edge index (COO format)

        Returns
        -------
        z : [N, out_channels]  projected node embeddings
        """
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return self.projection(x)
