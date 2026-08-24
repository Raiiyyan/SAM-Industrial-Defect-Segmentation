import torch
import torch.nn as nn


class IndustrialAdapter(nn.Module):
    def __init__(self, dim, bottleneck_dim=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.activation(x)
        x = self.up(x)
        return x + residual