# src/training/trainer.py

import copy
import torch
import torch.utils.data as Data

from utils.config import DEVICE, DTYPE
from training.losses import mse
from training.pcgrad import pcgrad_step
from physics.pde import pde_residual
from physics.beam_prior import beam_prior_loss, I_DEFAULT


def train_physics_only(
    model,
    xyn_t,
    xn_int_bank,
    fixed_t, zero_uv,
    right_t, top_t, bot_t,
    mid_rt, ds_rt,
    xn_slice, xn_midline,
    x_centers, y_line, Yc,
    x_min, x_max, y_min, y_max, y_mid,
    E, F_total,
    t_thick,
    sigma_plane_stress,
    I=I_DEFAULT,
    epochs=3000,
    lr=1e-3,
):
    """
    Physics-only training loop for the cantilever beam PINN with PCGrad.

    Returns
    -------
    best_state : dict
        Best model state_dict found during training.
    history : dict of lists
        Training history of different loss components.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.8, patience=150, threshold=1e-5
    )

    dataset = Data.TensorDataset(xyn_t)
    loader = Data.DataLoader(
        dataset,
        batch_size=min(4096, len(xyn_t)),
        shuffle=True,
    )

    L = x_max - x_min
    v_tip_target_theory = -F_total * L**3 / (3.0 * E * I)
    v_tip_target = torch.tensor(
        float(v_tip_target_theory), dtype=DTYPE, device=DEVICE
    )

    # Loss weights
    W_PDE      = 0.03
    W_BC       = 1.0
    W_RES      = 0.3
    W_FREE     = 0.1
    W_SYY      = 0.0

    W_NEUTRAL  = 0.01
    W_MIDLINE  = 0.0

    W_MOMENT   = 0.02
    W_BEAM     = 0.02
    W_UNIFORM  = 1.0

    W_TIP      = 200.0

    hist_total, hist_pde, hist_bc = [], [], []
    hist_res, hist_free, hist_syy = [], [], []
    hist_neutral, hist_mid, hist_moment = [], [], []
    hist_beam, hist_uniform, hist_tip = [], [], []

    best_loss = 1e99
    best_state = None

    n_bins = x_centers.shape[0]
    n_y    = y_line.shape[0]

    for epoch in range(epochs):
        model.train()
        ep_total = ep_pde = ep_bc = 0.0
        ep_res = ep_free = ep_syy = 0.0
        ep_neutral = ep_mid = ep_moment = 0.0
        ep_beam = ep_uniform = ep_tip = 0.0

        for (batch_xn,) in loader:
            # PDE on a mix of mesh nodes + interior samples
            k = min(4 * batch_xn.shape[0], xn_int_bank.shape[0])
            idx = torch.randint(0, xn_int_bank.shape[0], (k,), device=DEVICE)
            mix_xn = torch.cat([batch_xn, xn_int_bank[idx]], dim=0)

            # PDE residual in the bulk
            r1, r2, (sxx_i, syy_i, sxy_i), _ = pde_residual(
                mix_xn,
                model,
                x_min, x_max, y_min, y_max,
                sigma_plane_stress,
            )
            loss_pde = mse(r1, torch.zeros_like(r1)) + mse(
                r2, torch.zeros_like(r2)
            )

            # Plane-stress prior: sigma_yy ≈ 0 in the interior
            loss_syy = mse(syy_i, torch.zeros_like(syy_i))

            # Clamped BC: u = v = 0 on left edge (diagnostic)
            uv_fix  = model(fixed_t)
            loss_bc = mse(uv_fix, zero_uv)

            # Shear resultant on right edge = F_total (signed)
            _, _, (_, _, sxy_mid), _ = pde_residual(
                mid_rt,
                model,
                x_min, x_max, y_min, y_max,
                sigma_plane_stress,
            )
            F_pred   = torch.sum(sxy_mid * t_thick * ds_rt)
            loss_res = (F_pred + F_total)**2

            # Free top and bottom boundaries
            loss_free = 0.0
            if top_t is not None and len(top_t):
                _, _, (_, syy_top, sxy_top), _ = pde_residual(
                    top_t,
                    model,
                    x_min, x_max, y_min, y_max,
                    sigma_plane_stress,
                )
                loss_free += mse(syy_top, torch.zeros_like(syy_top)) \
                           + mse(sxy_top, torch.zeros_like(sxy_top))
            if bot_t is not None and len(bot_t):
                _, _, (_, syy_bot, sxy_bot), _ = pde_residual(
                    bot_t,
                    model,
                    x_min, x_max, y_min, y_max,
                    sigma_plane_stress,
                )
                loss_free += mse(syy_bot, torch.zeros_like(syy_bot)) \
                           + mse(sxy_bot, torch.zeros_like(sxy_bot))

            # Neutral axis: ∫ sigma_xx dy ≈ 0 per x-slice
            _, _, (sxx_slice, _, _), _ = pde_residual(
                xn_slice,
                model,
                x_min, x_max, y_min, y_max,
                sigma_plane_stress,
            )
            sxx_slice = sxx_slice.reshape(n_bins, n_y)
            loss_neutral_vec = torch.trapz(sxx_slice, y_line, dim=1)
            loss_neutral = torch.mean(loss_neutral_vec**2)

            # sigma_xx at mid-height ≈ 0
            _, _, (sxx_mid, _, _), _ = pde_residual(
                xn_midline,
                model,
                x_min, x_max, y_min, y_max,
                sigma_plane_stress,
            )
            loss_midline = mse(sxx_mid, torch.zeros_like(sxx_mid))

            # Bending moment per slice (signed)
            _, _, (sxx_slice_M, _, _), _ = pde_residual(
                xn_slice,
                model,
                x_min, x_max, y_min, y_max,
                sigma_plane_stress,
            )
            sxx_slice_M = sxx_slice_M.reshape(n_bins, n_y)
            M_pred_vec = torch.trapz(
                sxx_slice_M * (Yc - y_mid), y_line, dim=1
            ) * t_thick
            M_tgt_vec  = F_total * (x_max - x_centers)
            loss_moment = mse(M_pred_vec, M_tgt_vec)

            # Beam prior (v'' ~ M / (E I))
            loss_beam = beam_prior_loss(
                model,
                x_min, x_max, y_min, y_max,
                E, F_total,
                mse,
                I=I,
            )

            # Uniform transverse displacement on right edge
            uv_right = model(right_t)
            v_right  = uv_right[:, 1]
            loss_right_uniform = torch.var(v_right - v_right.mean())

            # Average tip deflection ~ analytical beam theory
            loss_tip = mse(v_right.mean(), v_tip_target)

            # Weighted losses for PCGrad
            L_pde     = W_PDE     * loss_pde
            L_bc      = W_BC      * loss_bc
            L_res     = W_RES     * loss_res
            L_free    = W_FREE    * loss_free
            L_syy     = W_SYY     * loss_syy
            L_neutral = W_NEUTRAL * loss_neutral
            L_mid     = W_MIDLINE * loss_midline
            L_moment  = W_MOMENT  * loss_moment
            L_beam    = W_BEAM    * loss_beam
            L_unif    = W_UNIFORM * loss_right_uniform
            L_tip     = W_TIP     * loss_tip

            loss_list = [
                L_pde, L_bc, L_res, L_free, L_syy,
                L_neutral, L_mid, L_moment, L_beam,
                L_unif, L_tip,
            ]

            # Scalar total for logging / scheduler
            loss_total_batch = sum(
                li.detach().cpu().item() for li in loss_list
            )

            # PCGrad step
            pcgrad_step(model, opt, loss_list)

            # Accumulate losses
            ep_total   += loss_total_batch
            ep_pde     += float(loss_pde.detach().cpu())
            ep_bc      += float(loss_bc.detach().cpu())
            ep_res     += float(loss_res.detach().cpu())
            ep_free    += float(loss_free.detach().cpu())
            ep_syy     += float(loss_syy.detach().cpu())
            ep_neutral += float(loss_neutral.detach().cpu())
            ep_mid     += float(loss_midline.detach().cpu())
            ep_moment  += float(loss_moment.detach().cpu())
            ep_beam    += float(loss_beam.detach().cpu())
            ep_uniform += float(loss_right_uniform.detach().cpu())
            ep_tip     += float(loss_tip.detach().cpu())

        sch.step(ep_total)

        hist_total.append(ep_total); hist_pde.append(ep_pde); hist_bc.append(ep_bc)
        hist_res.append(ep_res);     hist_free.append(ep_free); hist_syy.append(ep_syy)
        hist_neutral.append(ep_neutral); hist_mid.append(ep_mid); hist_moment.append(ep_moment)
        hist_beam.append(ep_beam);   hist_uniform.append(ep_uniform); hist_tip.append(ep_tip)

        if (epoch + 1) % 100 == 0:
            print(
                f"[{epoch+1:5d}] total={ep_total:.3e} | pde={ep_pde:.3e} | bc={ep_bc:.3e} "
                f"| res={ep_res:.3e} | free={ep_free:.3e} | syy={ep_syy:.3e} "
                f"| neutral={ep_neutral:.3e} | mid={ep_mid:.3e} | moment={ep_moment:.3e} "
                f"| beam={ep_beam:.3e} | unif={ep_uniform:.3e} | tip={ep_tip:.3e}"
            )

        if ep_total < best_loss:
            best_loss = ep_total
            best_state = copy.deepcopy(model.state_dict())

    history = {
        "total": hist_total,
        "pde": hist_pde,
        "bc": hist_bc,
        "res": hist_res,
        "free": hist_free,
        "syy": hist_syy,
        "neutral": hist_neutral,
        "mid": hist_mid,
        "moment": hist_moment,
        "beam": hist_beam,
        "uniform": hist_uniform,
        "tip": hist_tip,
    }

    return best_state, history, v_tip_target_theory


