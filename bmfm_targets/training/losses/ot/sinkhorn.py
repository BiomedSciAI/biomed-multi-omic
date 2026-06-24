"""
Log-domain Sinkhorn and debiased Sinkhorn divergence (pure PyTorch, fp32-internally).

All public functions upcast fp16/bfloat16 inputs to float32 internally; float64 inputs are
preserved to allow numerical gradient checks. Outputs always have at least float32 precision
(never fp16). This makes them safe to call with float16 inputs while still supporting gradcheck
in float64.

Assumptions
-----------
- Uniform weights: both marginals are 1/N uniform. The equal-mass correction term
  ``0.5 * eps * (sum_a - sum_b)^2`` in the debiased divergence is therefore always 0
  and is omitted.
- The debiased Sinkhorn divergence is:
      SD(X, Y) = OT_eps(X, Y) - 0.5*OT_eps(X, X) - 0.5*OT_eps(Y, Y)
  where OT_eps is the entropic regularized optimal-transport cost (``sinkhorn_cost``).
"""

from __future__ import annotations

from typing import Literal

import torch

CostName = Literal["euclidean", "sqeuclidean"]

_LOW_PRECISION = {torch.float16, torch.bfloat16}


def _safe_cast(x: torch.Tensor) -> torch.Tensor:
    """Upcast fp16/bf16 to fp32; preserve fp32 and fp64 as-is."""
    if x.dtype in _LOW_PRECISION:
        return x.float()
    return x


def _cost_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    cost: CostName = "euclidean",
) -> torch.Tensor:
    """
    Pairwise cost matrix C[i,j] = c(x_i, y_j).

    Parameters
    ----------
    x:
        Shape [N, d], float32.
    y:
        Shape [M, d], float32.
    cost:
        ``"euclidean"`` for ``||x-y||_2``; ``"sqeuclidean"`` for ``||x-y||_2^2``.

    Returns
    -------
    torch.Tensor
        Shape [N, M], float32.

    """
    C = torch.cdist(x, y, p=2)  # [N, M]
    if cost == "sqeuclidean":
        C = C**2
    return C


def _sinkhorn_log(
    C: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    n_iters: int,
) -> torch.Tensor:
    """
    Log-domain Sinkhorn returning the entropic OT cost (scalar).

    Parameters
    ----------
    C:
        Cost matrix [N, M], float32.
    a:
        Source weights [N], float32, positive, sums to 1.
    b:
        Target weights [M], float32, positive, sums to 1.
    eps:
        Regularization strength (>0).
    n_iters:
        Number of Sinkhorn iterations.

    Returns
    -------
    torch.Tensor
        Scalar: the regularized OT cost ``<pi*, C> + eps * KL(pi*||a*b)``.
        Equivalently the Sinkhorn entropic cost as used in OTT-JAX.

    """
    log_a = torch.log(a)  # [N]
    log_b = torch.log(b)  # [M]

    # M_ij = -(C_ij / eps)
    # log-sum-exp over rows/cols using logsumexp
    M = -C / eps  # [N, M]

    log_u = torch.zeros_like(log_a)  # [N]
    log_v = torch.zeros_like(log_b)  # [M]

    for _ in range(n_iters):
        # u update: log_u = log_a - logsumexp(M + log_v, dim=1)
        log_u = log_a - torch.logsumexp(M + log_v.unsqueeze(0), dim=1)
        # v update: log_v = log_b - logsumexp(M.T + log_u, dim=1)
        log_v = log_b - torch.logsumexp(M.t() + log_u.unsqueeze(0), dim=1)

    # log-transport plan: log_pi_ij = M_ij + log_u_i + log_v_j
    log_pi = M + log_u.unsqueeze(1) + log_v.unsqueeze(0)  # [N, M]

    # OT cost = sum_{i,j} pi_ij * C_ij + eps * sum_{i,j} pi_ij * (log_pi_ij - 1)
    # = sum_{i,j} pi_ij * C_ij + eps * KL(pi || a x b) - eps (the "- eps" vanishes for
    #   the "entropic" formulation used in OTT-JAX which returns:
    #   reg_ot_cost = <pi, C> + eps * sum_ij pi_ij * (log_pi_ij - log(a_i * b_j))
    # We use the standard ent_reg_cost form (same as monge_gap_from_samples in OTT-JAX):
    #   ent_reg_cost = sum_ij pi_ij * C_ij + eps * H_KL(pi || a*b)
    # where H_KL = sum pi_ij (log pi_ij - log a_i - log b_j)
    pi = log_pi.exp()
    transport_cost = (pi * C).sum()
    entropy_term = (pi * (log_pi - log_a.unsqueeze(1) - log_b.unsqueeze(0))).sum()
    ent_reg_cost = transport_cost + eps * entropy_term
    return ent_reg_cost


def sinkhorn_cost(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 1.0,
    n_iters: int = 100,
    cost: CostName = "euclidean",
) -> torch.Tensor:
    """
    Entropic regularized OT cost between point clouds X and Y.

    Inputs are upcasted to float32 internally. Uniform weights are assumed.

    Parameters
    ----------
    X:
        Shape [N, d].
    Y:
        Shape [M, d].
    eps:
        Entropic regularization (>0). Default 1.0.
    n_iters:
        Number of Sinkhorn iterations. Default 100.
    cost:
        ``"euclidean"`` or ``"sqeuclidean"``. Default ``"euclidean"``.

    Returns
    -------
    torch.Tensor
        Scalar (at least float32) entropic OT cost.

    """
    X = _safe_cast(X)
    Y = _safe_cast(Y)
    N = X.shape[0]
    M = Y.shape[0]
    device = X.device
    a = torch.full((N,), 1.0 / N, dtype=X.dtype, device=device)
    b = torch.full((M,), 1.0 / M, dtype=Y.dtype, device=device)
    C = _cost_matrix(X, Y, cost)
    return _sinkhorn_log(C, a, b, eps, n_iters)


def sinkhorn_divergence(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 1.0,
    n_iters: int = 100,
    cost: CostName = "euclidean",
) -> torch.Tensor:
    """
    Debiased Sinkhorn divergence between point clouds X and Y.

    ``SD(X, Y) = OT_eps(X, Y) - 0.5 * OT_eps(X, X) - 0.5 * OT_eps(Y, Y)``

    The equal-mass correction ``0.5 * eps * (sum_a - sum_b)^2`` is 0 for
    uniform weights and is omitted.

    Inputs are upcasted to float32. Returns float32 scalar.

    Parameters
    ----------
    X:
        Shape [N, d].
    Y:
        Shape [M, d].
    eps:
        Regularization. Default 1.0.
    n_iters:
        Sinkhorn iterations. Default 100.
    cost:
        Cost function. Default ``"euclidean"``.

    Returns
    -------
    torch.Tensor
        Scalar float32 debiased Sinkhorn divergence. Equals 0 when X == Y.

    """
    ot_xy = sinkhorn_cost(X, Y, eps=eps, n_iters=n_iters, cost=cost)
    ot_xx = sinkhorn_cost(X, X, eps=eps, n_iters=n_iters, cost=cost)
    ot_yy = sinkhorn_cost(Y, Y, eps=eps, n_iters=n_iters, cost=cost)
    return ot_xy - 0.5 * ot_xx - 0.5 * ot_yy
