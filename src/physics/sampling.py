# src/physics/sampling.py

import numpy as np
import torch
from utils.config import DEVICE, DTYPE
from physics.coordinates import phys_to_xn


def build_neutral_axis_samples(x_min, x_max, y_min, y_max, y_mid,
                               n_bins=24, n_y=21):
    """
    Build sampling points for neutral axis and bending-moment constraints.

    Returns
    -------
    samples : dict with
        x_centers : (n_bins,) tensor
        y_line    : (n_y,) tensor
        Xc, Yc    : (n_bins, n_y) tensors (meshgrid)
        xy_slice_phys : (n_bins * n_y, 2) tensor
        xn_slice      : (n_bins * n_y, 2) tensor
        x_midline     : (N_mid,) tensor
        xy_mid_phys   : (N_mid, 2) tensor
        xn_midline    : (N_mid, 2) tensor
    """
    # 1D lines in x and y
    x_centers = torch.linspace(x_min, x_max, n_bins,
                               dtype=DTYPE, device=DEVICE)
    y_line = torch.linspace(y_min, y_max, n_y,
                            dtype=DTYPE, device=DEVICE)

    # 2D grid (x, y)
    Xc, Yc = torch.meshgrid(x_centers, y_line, indexing="ij")

    xy_slice_phys = torch.stack(
        [Xc.reshape(-1), Yc.reshape(-1)], dim=1
    )

    # Map physical -> normalized
    xn_slice = phys_to_xn(xy_slice_phys, x_min, x_max, y_min, y_max).detach()

    # Midline at y = y_mid
    x_midline = torch.linspace(x_min, x_max, 128,
                               dtype=DTYPE, device=DEVICE)
    xy_mid_phys = torch.stack(
        [x_midline, torch.full_like(x_midline, y_mid)], dim=1
    )
    xn_midline = phys_to_xn(xy_mid_phys, x_min, x_max, y_min, y_max).detach()

    return {
        "x_centers": x_centers,
        "y_line": y_line,
        "Xc": Xc,
        "Yc": Yc,
        "xy_slice_phys": xy_slice_phys,
        "xn_slice": xn_slice,
        "x_midline": x_midline,
        "xy_mid_phys": xy_mid_phys,
        "xn_midline": xn_midline,
    }


def sample_interior(n, device=DEVICE, dtype=DTYPE):
    """
    Sample n interior collocation points in normalized space [-1, 1]^2.
    """
    xn = torch.rand(n, 2, device=device, dtype=dtype) * 2.0 - 1.0
    return xn


def build_interior_bank(n_nodes, min_points=20000,
                        device=DEVICE, dtype=DTYPE):
    """
    Build a bank of interior points, size = max(4 * n_nodes, min_points).
    """
    N_int_bank = max(4 * n_nodes, min_points)
    xn_int_bank = sample_interior(N_int_bank, device=device, dtype=dtype)
    return xn_int_bank
