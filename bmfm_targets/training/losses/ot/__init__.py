"""
Pure-torch Optimal Transport loss module for CMonge perturbation training.

Public API
----------
sinkhorn_cost
    Entropic regularized OT cost between two point clouds.
sinkhorn_divergence
    Debiased Sinkhorn divergence (SD(X,X) == 0).
monge_gap
    Monge gap regularizer (mean displacement minus entropic OT cost).
cmonge_fitting_and_regularizer
    Combined CMonge objective: fitting loss + Monge gap regularizer.
"""

from bmfm_targets.training.losses.ot.cmonge_loss import cmonge_fitting_and_regularizer
from bmfm_targets.training.losses.ot.monge_gap import monge_gap
from bmfm_targets.training.losses.ot.sinkhorn import sinkhorn_cost, sinkhorn_divergence

__all__ = [
    "sinkhorn_cost",
    "sinkhorn_divergence",
    "monge_gap",
    "cmonge_fitting_and_regularizer",
]
