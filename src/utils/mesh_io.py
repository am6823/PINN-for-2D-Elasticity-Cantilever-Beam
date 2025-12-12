# src/utils/mesh_io.py

import re
import numpy as np


def load_nodes_from_fem(path):
    """
    Read node IDs and (x, y) coordinates from a Nastran-like .fem file.

    Parameters
    ----------
    path : str
        Path to the .fem file.

    Returns
    -------
    nodes_xyz : np.ndarray of shape (N, 3)
        Array with columns [node_id, x, y].
    """
    nodes = []
    with open(path, "r") as f:
        for line in f:
            if line.strip().startswith("GRID"):
                vals = re.findall(r"[-+]?\d*\.?\d+", line)
                if len(vals) >= 4:
                    node_id, x, y, z = map(float, vals[:4])
                    nodes.append([int(node_id), x, y])

    nodes_xyz = np.array(nodes, dtype=float)
    return nodes_xyz
