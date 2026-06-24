"""Samplers for condition-homogeneous mini-batching (CMonge port)."""

import random
from collections import defaultdict

import pandas as pd
from torch.utils.data import Sampler


class ConditionHomogeneousBatchSampler(Sampler):
    """
    Yields batches of perturbed-cell indices where every index in a batch
    shares the same perturbation condition.

    At each epoch step one condition is sampled uniformly at random; then
    batch_size indices are drawn *with replacement* from that condition's
    index pool.  This makes every batch condition-homogeneous so that the
    within-batch population comparison needed by CMonge is valid.

    Controls are paired inside BasePerturbationDataset.__getitem__, so we
    only track perturbed-cell indices here.

    Parameters
    ----------
    obs_conditions : pd.Series
        A pandas Series of length == len(perturbed cells) whose values are
        the condition label for each perturbed cell.  Typically obtained as
        ``dataset.perturbation_cells.obs[dataset.perturbation_column_name]``.
    batch_size : int
        Number of indices per yielded batch (all same condition).
    num_batches : int | None
        Number of batches per epoch.  Defaults to
        ``len(unique_conditions) * ceil(max_cells_per_condition / batch_size)``
        — approximately one pass through every cell.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        obs_conditions: pd.Series,
        batch_size: int,
        num_batches: int | None = None,
        seed: int | None = None,
    ):
        import math

        self.batch_size = batch_size
        self.seed = seed

        # Build condition -> list[int] mapping (iloc positions)
        condition_to_indices: dict[str, list[int]] = defaultdict(list)
        for iloc_pos, cond in enumerate(obs_conditions):
            condition_to_indices[cond].append(iloc_pos)
        self._condition_to_indices = dict(condition_to_indices)
        self._conditions = sorted(self._condition_to_indices.keys())

        if num_batches is None:
            max_cells = max(len(v) for v in self._condition_to_indices.values())
            num_batches = len(self._conditions) * math.ceil(max_cells / batch_size)
        self._num_batches = num_batches

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self._num_batches):
            # Uniformly sample one condition
            cond = rng.choice(self._conditions)
            pool = self._condition_to_indices[cond]
            # Sample batch_size indices with replacement
            batch = rng.choices(pool, k=self.batch_size)
            yield batch
