"""Two-layer GATv2 residual model used by every NMGAT experiment."""
import torch
from torch import nn
from torch_geometric.nn.models import GAT

class MassNet(nn.Module):
    """2-layer GATv2 for node-level regression (delta/residual prediction)."""

    def __init__(
        self,
        in_dim: int = 17,
        hidden_dim: int = 64,
        out_dim: int = 1,
        heads: int = 2,
        dropout: float = 0.2,
        concat: bool = False,
        norm: str | None = "layer_norm",
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

        self.gnn = GAT(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=2,
            dropout=dropout,
            act="leaky_relu",
            act_kwargs={"negative_slope": 0.2},
            norm=norm,
            v2=True,
            heads=heads,
            concat=concat,
            residual=False,
        )

        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.gnn(h, edge_index)
        return self.head(h).view(-1)
