# src/evaluation/prediction.py

import torch
from physics.coordinates import xn_to_phys, phys_to_xn


def predict_uv_sigma(xn, model,
                     x_min, x_max, y_min, y_max,
                     sigma_plane_stress):
    """
    Compute displacement and stresses at given normalized coordinates.

    Parameters
    ----------
    xn : (N, 2) tensor
        Normalized coordinates in [-1, 1]^2.
    model : nn.Module
        Trained PINN model.
    x_min, x_max, y_min, y_max : float
        Physical bounds of the domain.
    sigma_plane_stress : callable
        Constitutive law: sigma_plane_stress(eps_xx, eps_yy, gamma_xy).

    Returns
    -------
    uv : (N, 2) tensor
        Displacements [u, v].
    sxx, syy, sxy : tensors
        Stress components at each point.
    """
    # Physical coordinates with gradients enabled
    x_phys = xn_to_phys(xn, x_min, x_max, y_min, y_max)
    x_phys_req = x_phys.detach().requires_grad_(True)

    # Back to normalized space for the model
    xn_from_phys = phys_to_xn(x_phys_req, x_min, x_max, y_min, y_max)
    uv = model(xn_from_phys)

    u = uv[:, 0]
    v = uv[:, 1]

    du = torch.autograd.grad(
        u, x_phys_req, torch.ones_like(u),
        create_graph=False, retain_graph=True
    )[0]
    dv = torch.autograd.grad(
        v, x_phys_req, torch.ones_like(v),
        create_graph=False, retain_graph=False
    )[0]

    du_dx, du_dy = du[:, 0], du[:, 1]
    dv_dx, dv_dy = dv[:, 0], dv[:, 1]

    eps_xx   = du_dx
    eps_yy   = dv_dy
    gamma_xy = du_dy + dv_dx

    sxx, syy, sxy = sigma_plane_stress(eps_xx, eps_yy, gamma_xy)
    return uv, sxx, syy, sxy

