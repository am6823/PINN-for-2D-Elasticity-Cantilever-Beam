# src/physics/geometry.py

import numpy as np


def init_geometry(nodes_xyz, tol=1e-9):
    """
    Build all geometric quantities from nodes_xyz = [[id, x, y], ...].

    Parameters
    ----------
    nodes_xyz : np.ndarray of shape (N, 3)
        Node IDs and (x, y) coordinates.

    Returns
    -------
    geom : dict
        Dictionary with geometry-related arrays and scalars.
    """
    ids = nodes_xyz[:, 0].astype(int)
    xy = nodes_xyz[:, 1:3].astype(float)

    # Bounds
    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    y_min, y_max = xy[:, 1].min(), xy[:, 1].max()
    y_mid = 0.5 * (y_min + y_max)

    # Beam dimensions
    L = x_max - x_min
    h = y_max - y_min

    # Shifted coordinates for plotting (x in [0, L], y in [0, h])
    xy_plot = xy.copy()
    xy_plot[:, 0] = xy[:, 0] - x_min
    xy_plot[:, 1] = xy[:, 1] - y_min

    # Left (clamped) and right (loaded) edges
    left_xy = xy[np.isclose(xy[:, 0], x_min, atol=tol)]
    load_mask = np.isclose(xy[:, 0], x_max, atol=tol)
    right_xy = xy[load_mask]

    # Top and bottom edges
    top_xy = xy[np.isclose(xy[:, 1], y_max, atol=tol)]
    bot_xy = xy[np.isclose(xy[:, 1], y_min, atol=tol)]

    geom = {
        "ids": ids,
        "xy": xy,
        "xy_plot": xy_plot,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "y_mid": y_mid,
        "L": L,
        "h": h,
        "tol": tol,
        "left_xy": left_xy,
        "right_xy": right_xy,
        "load_mask": load_mask,
        "top_xy": top_xy,
        "bot_xy": bot_xy,
    }
    return geom

