# src/physics/pde.py

import torch
from physics.coordinates import xn_to_phys, phys_to_xn
from utils.autograd import grads_scalar_wrt_phys, compute_stress_autograd


def pde_residual(xn, model,
                 x_min, x_max, y_min, y_max,
                 sigma_plane_stress):
    """
    PDE residual in the bulk for 2D linear elasticity:
        div(sigma) = 0.

    Parameters
    ----------
    xn : (N, 2) tensor
        Normalized coordinates in [-1, 1]^2.
    model : nn.Module
        PINN model mapping normalized coords -> [u, v].
    x_min, x_max, y_min, y_max : float
        Bounds of the physical domain.
    sigma_plane_stress : callable
        Function sigma_plane_stress(eps_xx, eps_yy, gamma_xy)
        returning (sig_xx, sig_yy, sig_xy).

    Returns
    -------
    r1, r2 : tensors
        PDE residual components.
    (sxx, syy, sxy) : tuple of tensors
        Stresses at the collocation points.
    uv : (N, 2) tensor
        Displacement field at the collocation points.
    """
    # Map normalized coords to physical space and make them differentiable
    x_phys = xn_to_phys(xn, x_min, x_max, y_min, y_max)
    x_phys = x_phys.detach().requires_grad_(True)

    # Map back to normalized coords for the model
    xn_from_phys = phys_to_xn(x_phys, x_min, x_max, y_min, y_max)

    # Forward pass
    uv = model(xn_from_phys)

    # Stresses from displacement field
    sxx, syy, sxy, _, _ = compute_stress_autograd(uv, x_phys, sigma_plane_stress)

    # Divergence of sigma
    dsxx_dx, _ = grads_scalar_wrt_phys(sxx, x_phys)
    dsxy_dx, dsxy_dy = grads_scalar_wrt_phys(sxy, x_phys)
    _, dsyy_dy = grads_scalar_wrt_phys(syy, x_phys)

    r1 = dsxx_dx + dsxy_dy
    r2 = dsxy_dx + dsyy_dy

    return r1, r2, (sxx, syy, sxy), uv
