"""
Monge gap regularizer (pure PyTorch, fp32-internally).

Monge gap measures how far a map T (source -> mapped) is from an optimal transport map.
It is defined as:

    monge_gap(source, mapped) = mean_i c(source_i, mapped_i) - OT_eps(source, mapped)

where OT_eps is the entropic regularized optimal-transport cost (``sinkhorn_cost``).

This quantity is >= 0; minimizing it pushes the map T toward an optimal (efficient)
transport. Reference: Uscidda & Cuturi, "The Monge Gap: A Regularizer to Learn
All Transport Maps", ICML 2023.

Note on sign: RECON.md §A corrects the plan. The formula here (mean displacement MINUS OT)
matches the OTT-JAX reference implementation ``monge_gap_from_samples``.
"""

from __future__ import annotations

import torch

from bmfm_targets.training.losses.ot.sinkhorn import (
    CostName,
    _cost_matrix,
    _safe_cast,
    sinkhorn_cost,
)


def monge_gap(
    source: torch.Tensor,
    mapped: torch.Tensor,
    eps: float = 0.01,
    n_iters: int = 100,
    cost: CostName = "euclidean",
) -> torch.Tensor:
    """
    Monge gap between source and mapped point clouds.

    ``monge_gap = mean_i c(source_i, mapped_i) - OT_eps(source, mapped)``

    Both tensors must have the same shape ``[N, d]``. fp16/bf16 inputs are upcasted
    to fp32; fp64 inputs are preserved to allow gradcheck.

    Parameters
    ----------
    source:
        Shape [N, d]. The source points (e.g. control cell embeddings).
    mapped:
        Shape [N, d]. The mapped / transported points (e.g. predicted perturbation).
    eps:
        Entropic regularization for the OT baseline. Default 0.01.
    n_iters:
        Sinkhorn iterations. Default 100.
    cost:
        Cost function: ``"euclidean"`` or ``"sqeuclidean"``. Default ``"euclidean"``.

    Returns
    -------
    torch.Tensor
        Scalar (at least float32) Monge gap (>= 0).

    """
    source = _safe_cast(source)
    mapped = _safe_cast(mapped)
    # mean_i c(source_i, mapped_i) — the diagonal of the pairwise cost matrix
    C_diag = _cost_matrix(source, mapped, cost).diag()  # [N]
    mean_disp = C_diag.mean()
    ot = sinkhorn_cost(source, mapped, eps=eps, n_iters=n_iters, cost=cost)
    return mean_disp - ot
