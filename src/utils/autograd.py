# src/utils/autograd.py

import torch


def grads_scalar_wrt_phys(f, x_phys,
                          create_graph=True,
                          retain_graph=True):
    """
    Gradient of a scalar field f with respect to physical coordinates (x, y).

    Parameters
    ----------
    f : (N,) tensor
        Scalar field evaluated at x_phys.
    x_phys : (N, 2) tensor
        Physical coordinates with requires_grad=True.
    create_graph, retain_graph : bool
        Flags passed to torch.autograd.grad.

    Returns
    -------
    df_dx, df_dy : tensors of shape (N,)
    """
    g = torch.autograd.grad(
        f,
        x_phys,
        grad_outputs=torch.ones_like(f),
        create_graph=create_graph,
        retain_graph=retain_graph,
    )[0]
    return g[:, 0], g[:, 1]


def compute_stress_autograd(uv, x_phys, sigma_plane_stress):
    """
    Compute strains and stresses from a displacement field via autograd.

    Parameters
    ----------
    uv : (N, 2) tensor
        Displacement field [u, v].
    x_phys : (N, 2) tensor
        Physical coordinates with requires_grad=True.
    sigma_plane_stress : callable
        Function sigma_plane_stress(eps_xx, eps_yy, gamma_xy)
        returning (sig_xx, sig_yy, sig_xy).

    Returns
    -------
    sig_xx, sig_yy, sig_xy, u, v : tensors
    """
    from utils.autograd import grads_scalar_wrt_phys  # to avoid circular import

    u = uv[:, 0]
    v = uv[:, 1]

    du_dx, du_dy = grads_scalar_wrt_phys(u, x_phys)
    dv_dx, dv_dy = grads_scalar_wrt_phys(v, x_phys)

    eps_xx = du_dx
    eps_yy = dv_dy
    gamma_xy = du_dy + dv_dx

    sig_xx, sig_yy, sig_xy = sigma_plane_stress(eps_xx, eps_yy, gamma_xy)
    return sig_xx, sig_yy, sig_xy, u, v
