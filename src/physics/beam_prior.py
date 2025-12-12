# src/physics/beam_prior.py

import torch
from utils.config import DEVICE, DTYPE
from physics.coordinates import xn_to_phys, phys_to_xn


# Second moment of area (mm^4), from FEA section properties
I_DEFAULT = 2.083e4  # mm^4


def beam_prior_loss(model,
                    x_min, x_max, y_min, y_max,
                    E, F_total,
                    mse,
                    I=I_DEFAULT,
                    n_samp=2048,
                    device=DEVICE,
                    dtype=DTYPE):
    """
    Beam prior enforcing v''(x) ≈ M(x) / (E I) in a distributed sense.

    Parameters
    ----------
    model : nn.Module
        PINN model mapping normalized coordinates -> [u, v].
    x_min, x_max, y_min, y_max : float
        Physical domain bounds.
    E : float
        Young's modulus (MPa).
    F_total : float
        Signed total load (N), negative for downward.
    mse : callable
        MSE loss function (e.g., training.losses.mse).
    I : float
        Second moment of area (mm^4).
    n_samp : int
        Number of random samples in [-1, 1]^2.
    device, dtype :
        Torch device and dtype.

    Returns
    -------
    loss_beam : scalar tensor
        Beam prior loss.
    """
    # Sample points in normalized space
    xn = torch.rand(n_samp, 2, device=device, dtype=dtype) * 2.0 - 1.0

    # Map to physical space and enable gradients
    x_phys = xn_to_phys(xn, x_min, x_max, y_min, y_max)
    x_phys = x_phys.detach().requires_grad_(True)

    # Map back to normalized coordinates for the model
    xn_back = phys_to_xn(x_phys, x_min, x_max, y_min, y_max)

    # Vertical displacement
    v = model(xn_back)[:, 1]

    # First derivative dv/dx
    dv = torch.autograd.grad(
        v,
        x_phys,
        torch.ones_like(v),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0]

    # Second derivative d2v/dx2
    d2v = torch.autograd.grad(
        dv,
        x_phys,
        torch.ones_like(dv),
        create_graph=False,
        retain_graph=False,
    )[0][:, 0]

    x = x_phys[:, 0]

    # Internal bending moment along the span (same sign as F_total)
    Mx = -F_total * (x_max - x)

    return mse(d2v, Mx / (E * I + 1e-30))
