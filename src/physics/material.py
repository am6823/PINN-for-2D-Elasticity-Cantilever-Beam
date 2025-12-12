# src/physics/material.py

# Plane-stress material parameters and global load for the beam benchmark.

E_DEFAULT = 55000.0      # MPa
NU_DEFAULT = 0.26
T_THICK_DEFAULT = 2.0    # mm

# Signed total force: negative = downward (y axis pointing up)
F_TOTAL_DEFAULT = -120.0  # N


def get_default_plane_stress():
    """
    Return (E, nu, t_thick, F_total) for the default beam benchmark.
    """
    return E_DEFAULT, NU_DEFAULT, T_THICK_DEFAULT, F_TOTAL_DEFAULT
