# src/training/pcgrad.py

import torch


def pcgrad_step(model, optimizer, loss_list):
    """
    Perform one optimization step using PCGrad on a list of scalar losses.

    Parameters
    ----------
    model : nn.Module
        Model with parameters to update.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    loss_list : list of scalar tensors
        Each entry is a (possibly weighted) loss term.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    n_tasks = len(loss_list)

    grads = []
    for i, li in enumerate(loss_list):
        retain = (i != n_tasks - 1)
        g = torch.autograd.grad(
            li,
            params,
            retain_graph=retain,
            create_graph=False,
            allow_unused=True,
        )

        # Replace None (unused params) with zeros
        g_filled = []
        for p, g_ij in zip(params, g):
            if g_ij is None:
                g_filled.append(torch.zeros_like(p))
            else:
                g_filled.append(g_ij)
        grads.append(g_filled)

    # Copy gradients for projection
    pc_grads = [[g_ij.clone() for g_ij in g_i] for g_i in grads]

    eps = 1e-12

    # Project conflicts
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i == j:
                continue
            dot_ij = sum((pc_grads[i][k] * pc_grads[j][k]).sum()
                         for k in range(len(params)))
            if dot_ij < 0:
                norm_j_sq = sum((pc_grads[j][k]**2).sum()
                                for k in range(len(params))) + eps
                alpha = dot_ij / norm_j_sq
                for k in range(len(params)):
                    pc_grads[i][k] = pc_grads[i][k] - alpha * pc_grads[j][k]

    # Average projected gradients and step
    optimizer.zero_grad()
    for p_idx, p in enumerate(params):
        g_sum = sum(pc_grads[t][p_idx] for t in range(n_tasks))
        p.grad = g_sum / float(n_tasks)
    optimizer.step()
