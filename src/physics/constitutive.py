# src/physics/constitutive.py

def make_sigma_plane_stress(E, nu):
    """
    Build a plane-stress Hooke law sigma(eps) with given (E, nu).

    Returns
    -------
    sigma_plane_stress : callable
        Function taking (eps_xx, eps_yy, gamma_xy) and returning
        (sigma_xx, sigma_yy, sigma_xy).
    """
    coef = E / (1.0 - nu**2)
    G = E / (2.0 * (1.0 + nu))

    def sigma_plane_stress(eps_xx, eps_yy, gamma_xy):
        sig_xx = coef * (eps_xx + nu * eps_yy)
        sig_yy = coef * (nu * eps_xx + eps_yy)
        sig_xy = G * gamma_xy
        return sig_xx, sig_yy, sig_xy

    return sigma_plane_stress
