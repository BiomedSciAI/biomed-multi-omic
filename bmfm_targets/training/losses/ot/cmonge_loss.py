"""
Conditional Monge Gap combined loss (pure PyTorch, fp32-internally).

The CMonge objective combines:
1. A **fitting loss**: debiased Sinkhorn divergence between the target (true perturbed)
   population and the mapped (predicted perturbed) population.
2. A **Monge gap regularizer**: measures how far the map source -> mapped deviates from
   an optimal transport map.

Total loss:
    L = SD(target, mapped; eps_fitting) + reg_strength * monge_gap(source, mapped; eps_reg)

Reference: RECON.md §A; metrics.py in the JAX reference implementation.
"""

from __future__ import annotations

import torch

from bmfm_targets.training.losses.ot.monge_gap import monge_gap
from bmfm_targets.training.losses.ot.sinkhorn import CostName, sinkhorn_divergence


def cmonge_fitting_and_regularizer(
    source: torch.Tensor,
    target: torch.Tensor,
    mapped: torch.Tensor,
    eps_fitting: float = 1.0,
    eps_reg: float = 0.01,
    reg_strength: float = 1.0,
    cost: CostName = "euclidean",
    n_iters: int = 100,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the CMonge fitting loss and Monge gap regularizer.

    All inputs are upcasted to float32 internally.

    Parameters
    ----------
    source:
        Shape [N, d]. Control cell embeddings (encoder output before conditioning).
    target:
        Shape [M, d]. True perturbed cell observations (or expression vectors).
    mapped:
        Shape [N, d]. Predicted perturbed output (encoder output after conditioning /
        decoded expression predictions).
    eps_fitting:
        Regularization for the Sinkhorn divergence fitting loss. Default 1.0.
    eps_reg:
        Regularization for the Monge gap OT baseline. Default 0.01.
    reg_strength:
        Weight of the Monge gap regularizer. Default 1.0.
    cost:
        Cost function: ``"euclidean"`` (default, matches reference) or
        ``"sqeuclidean"``.
    n_iters:
        Number of Sinkhorn iterations. Default 100.

    Returns
    -------
    tuple[torch.Tensor, dict[str, torch.Tensor]]
        ``(total_loss, {"fitting": fitting_loss, "monge_gap": monge_gap_value})``
        where ``total_loss = fitting_loss + reg_strength * monge_gap_value``.

    """
    fitting = sinkhorn_divergence(
        target, mapped, eps=eps_fitting, n_iters=n_iters, cost=cost
    )
    mg = monge_gap(source, mapped, eps=eps_reg, n_iters=n_iters, cost=cost)
    total = fitting + reg_strength * mg
    return total, {"fitting": fitting, "monge_gap": mg}
