# src/physics/coordinates.py

import numpy as np
import torch
from utils.config import DEVICE, DTYPE


def normalize_xy(xy, x_min, x_max, y_min, y_max):
    """
    Normalize physical coordinates xy to [-1, 1]^2.
    """
    xn = 2.0 * (xy[:, 0] - x_min) / (x_max - x_min + 1e-12) - 1.0
    yn = 2.0 * (xy[:, 1] - y_min) / (y_max - y_min + 1e-12) - 1.0
    return np.stack([xn, yn], axis=1)


def xy_to_torch_normalized(xy, x_min, x_max, y_min, y_max, device=DEVICE, dtype=DTYPE):
    """
    Convenience helper: normalize xy and convert to a torch tensor.
    """
    xyn = normalize_xy(xy, x_min, x_max, y_min, y_max)
    return torch.tensor(xyn, device=device, dtype=dtype)

import numpy as np
import torch
from utils.config import DEVICE, DTYPE


def normalize_xy(xy, x_min, x_max, y_min, y_max):
    """
    Normalize physical coordinates xy to [-1, 1]^2.
    """
    xn = 2.0 * (xy[:, 0] - x_min) / (x_max - x_min + 1e-12) - 1.0
    yn = 2.0 * (xy[:, 1] - y_min) / (y_max - y_min + 1e-12) - 1.0
    return np.stack([xn, yn], axis=1)


def xy_to_torch_normalized(xy, x_min, x_max, y_min, y_max, device=DEVICE, dtype=DTYPE):
    """
    Convenience helper: normalize xy and convert to a torch tensor.
    """
    xyn = normalize_xy(xy, x_min, x_max, y_min, y_max)
    return torch.tensor(xyn, device=device, dtype=dtype)
