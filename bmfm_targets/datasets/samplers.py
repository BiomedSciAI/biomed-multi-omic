"""Samplers for condition-homogeneous mini-batching."""

import random
from collections import defaultdict

import pandas as pd
from torch.utils.data import Sampler


class ConditionHomogeneousBatchSampler(Sampler):
    """
    Yields batches of dataset indices where every index in a batch shares the
    same condition label (e.g. celltype/tissue).

    At each step one condition is sampled uniformly at random; then
    ``batch_size`` indices are drawn *with replacement* from that condition's
    index pool. Drawing homogeneous batches is required so the within-batch
    population can be compared against the reference population for that
    condition (e.g. the population-level OT loss).

    Because sampling is with replacement against a fixed ``num_batches``, an
    epoch is an approximate (not exhaustive) pass over the data; indices may be
    repeated and some may be skipped.

    Parameters
    ----------
    obs_conditions : pd.Series
        A pandas Series of length ``len(dataset)`` whose values are the
        condition label for each sample (e.g. the ``celltype_column`` of the
        dataset metadata).
    batch_size : int
        Number of indices per yielded batch (all the same condition).
    num_batches : int | None
        Number of batches per epoch. Defaults to
        ``len(unique_conditions) * ceil(max_cells_per_condition / batch_size)``
        — approximately one pass through every sample.
    seed : int | None
        Random seed for reproducibility. ``None`` (the default) means each
        epoch draws a fresh random sequence.
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
