"""
Convergence test for the Sinkhorn divergence loss.

Directly optimizes a small batch of predicted gene vectors against a fixed
target chip population.  No model, no data pipeline — pure loss convergence.

Pass criterion: OT loss must drop to <10% of its initial value within 200 steps.

Usage:
    PYTHONPATH=/u/dmichael/scRNA2ChIP python run/test_ot_convergence.py
"""

import torch

from bmfm_targets.training.losses.ot.sinkhorn import sinkhorn_divergence

torch.manual_seed(0)

N_GENES = 64  # small so Sinkhorn is fast
B = 8  # predicted batch size
M = 10  # chip population size
N_STEPS = 200
LR = 0.05

# Fixed target: all chip cells identical at [1, 0, 0, ..., 0] (one-hot on gene 0)
chip = torch.zeros(M, N_GENES)
chip[:, 0] = 1.0

# Predicted population: random init, to be optimised toward chip
pred = torch.randn(B, N_GENES, requires_grad=True)
optimizer = torch.optim.Adam([pred], lr=LR)

initial_loss = None
for step in range(N_STEPS):
    optimizer.zero_grad()
    loss = sinkhorn_divergence(pred, chip, eps=0.5, n_iters=50, cost="euclidean")
    loss.backward()
    optimizer.step()
    if step == 0:
        initial_loss = loss.item()
        print(f"step {step:3d}  loss={loss.item():.4f}  (initial)")
    elif step % 40 == 39:
        print(f"step {step:3d}  loss={loss.item():.4f}")

final_loss = loss.item()
ratio = final_loss / initial_loss
print(f"\ninitial={initial_loss:.4f}  final={final_loss:.4f}  ratio={ratio:.3f}")
assert (
    ratio < 0.10
), f"OT loss did not converge: final/initial = {ratio:.3f} (need <0.10)"
print("Convergence test PASSED")
