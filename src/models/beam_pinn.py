# src/models/beam_pinn.py

import torch
import torch.nn as nn
from utils.config import DEVICE, DTYPE


class BeamPINN(nn.Module):
    """
    Simple MLP for 2D elasticity PINN with hard-clamped BC on the left edge.

    Input:
        x : normalized coordinates in [-1, 1]^2
             x[:, 0] = -1 on the clamped left edge.
    Output:
        uv : displacement field [u, v] with u = v = 0 at x_n = -1 enforced
             by a multiplicative factor.
    """

    def __init__(self, sizes=(2, 64, 128, 64, 2)):
        """
        Parameters
        ----------
        sizes : tuple or list of int
            Layer sizes, e.g. (2, 64, 128, 64, 2).
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )

        # Xavier init for weights, zeros for biases
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.act = nn.SiLU()

        # Move model to global DEVICE/DTYPE
        self.to(device=DEVICE, dtype=DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hard clamped BC at x_n = -1.

        Parameters
        ----------
        x : (N, 2) tensor
            Normalized coordinates in [-1, 1]^2.

        Returns
        -------
        uv : (N, 2) tensor
            Displacements [u, v], with u = v = 0 on the clamped left edge.
        """
        z = x
        for layer in self.layers[:-1]:
            z = self.act(layer(z))

        # Unconstrained [u, v]
        uv_raw = self.layers[-1](z)

        # Enforce clamped BC exactly: u = v = 0 for x_n = -1
        s = (x[:, 0] + 1.0) / 2.0  # s = 0 at x_n = -1, s = 1 at x_n = +1
        s = s.unsqueeze(1)         # (N, 1), will broadcast over [u, v]

        uv = s * uv_raw
        return uv
