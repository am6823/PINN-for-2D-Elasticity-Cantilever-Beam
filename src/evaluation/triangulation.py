# src/evaluation/triangulation.py

from matplotlib.tri import Triangulation


def build_triangulation(xy_plot):
    """
    Build a Matplotlib Triangulation from shifted coordinates xy_plot.
    Returns None if triangulation fails.
    """
    try:
        tri = Triangulation(xy_plot[:, 0], xy_plot[:, 1])
    except Exception:
        tri = None
    return tri
