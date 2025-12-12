# src/physics/boundaries.py

import numpy as np
import torch
from utils.config import DEVICE, DTYPE
from physics.coordinates import normalize_xy


def build_boundary_sets(xy, left_xy, right_xy, x_min, x_max, y_min, y_max, tol=1e-9):
    """
    Build all boundary coordinate sets needed for the PINN:
        - fixed_t : clamped BC at left edge
        - right_t : points on right edge
        - top_t, bot_t : free top/bottom edges (may be None)
        - mid_rt, ds_rt : midpoints and segment lengths for integral resultant

    Parameters
    ----------
    xy : (N, 2) float ndarray
        All node coordinates.
    left_xy, right_xy : (M, 2) ndarrays
        Points on the left and right boundaries.
    x_min, x_max, y_min, y_max : float
        Domain bounds used for normalization.
    tol : float
        Tolerance for identifying top/bottom boundaries.

    Returns
    -------
    dict containing:
        fixed_t : (N_left, 2) tensor
        zero_uv : (N_left, 2) tensor of zeros
        right_t : (N_right, 2) tensor
        top_t : tensor or None
        bot_t : tensor or None
        mid_rt : (N_mid, 2) tensor
        ds_rt : (N_mid,) tensor
    """

    # --- Left (clamped) boundary ---
    fixed_norm = normalize_xy(left_xy, x_min, x_max, y_min, y_max)
    fixed_t = torch.tensor(fixed_norm, device=DEVICE, dtype=DTYPE)
    zero_uv = torch.zeros((len(fixed_t), 2), device=DEVICE, dtype=DTYPE)

    # --- Right boundary nodes ---
    right_norm = normalize_xy(right_xy, x_min, x_max, y_min, y_max)
    right_t = torch.tensor(right_norm, device=DEVICE, dtype=DTYPE)

    # --- Top and bottom boundaries ---
    top_xy = xy[np.isclose(xy[:, 1], y_max, atol=tol)]
    bot_xy = xy[np.isclose(xy[:, 1], y_min, atol=tol)]

    top_t = None
    bot_t = None

    if len(top_xy) > 0:
        tnorm = normalize_xy(top_xy, x_min, x_max, y_min, y_max)
        top_t = torch.tensor(tnorm, device=DEVICE, dtype=DTYPE)

    if len(bot_xy) > 0:
        bnorm = normalize_xy(bot_xy, x_min, x_max, y_min, y_max)
        bot_t = torch.tensor(bnorm, device=DEVICE, dtype=DTYPE)

    # --- Midpoints + arc-lengths on the right edge ---
    order = np.argsort(right_xy[:, 1])  # sort by y-coordinate
    pts_r = right_xy[order]

    if len(pts_r) < 2:
        raise RuntimeError("Right edge has too few nodes for integral constraint.")

    mid_r = 0.5 * (pts_r[1:] + pts_r[:-1])
    ds_r = np.linalg.norm(pts_r[1:] - pts_r[:-1], axis=1)

    mid_norm = normalize_xy(mid_r, x_min, x_max, y_min, y_max)
    mid_rt = torch.tensor(mid_norm, device=DEVICE, dtype=DTYPE)
    ds_rt = torch.tensor(ds_r, device=DEVICE, dtype=DTYPE)

    return {
        "fixed_t": fixed_t,
        "zero_uv": zero_uv,
        "right_t": right_t,
        "top_t": top_t,
        "bot_t": bot_t,
        "mid_rt": mid_rt,
        "ds_rt": ds_rt,
    }
