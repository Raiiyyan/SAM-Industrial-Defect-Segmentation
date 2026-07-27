"""
IndustrialAdapter — bottleneck adapter module injected into frozen SAM transformer blocks.

Architecture:  Linear(dim → bottleneck) → GELU → Linear(bottleneck → dim) + skip connection

The up-projection is initialized to near-zero so the adapter starts as a near-identity
function, keeping training stable from step 1.
"""

import torch
import torch.nn as nn


class IndustrialAdapter(nn.Module):
    """Bottleneck adapter with residual connection.

    Parameters
    ----------
    dim : int
        Input/output feature dimension (768 for SAM ViT-B).
    bottleneck_dim : int
        Compressed dimension in the bottleneck (default 64).
    """

    def __init__(self, dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, dim)

        # Near-zero init: up-projection starts as all-zeros so adapter(x) ≈ x
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down(x)
        x = self.activation(x)
        x = self.up(x)
        return x + residual
