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
    ``batch_size`` indices are drawn from that condition's index pool.
    When ``replacement=True`` (the default), indices are drawn with replacement.
    When ``replacement=False``, indices are drawn without replacement; if the
    pool is smaller than ``batch_size``, the pool is repeat-padded to fill the
    batch.

    Because sampling is against a fixed ``num_batches``, an epoch is an
    approximate (not exhaustive) pass over the data; indices may be repeated
    and some may be skipped.

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
    replacement : bool
        If ``True`` (default), draw indices with replacement. If ``False``,
        draw without replacement; repeat-pad if the pool is smaller than
        ``batch_size``.
    """

    def __init__(
        self,
        obs_conditions: pd.Series,
        batch_size: int,
        num_batches: int | None = None,
        seed: int | None = None,
        replacement: bool = False,
    ):
        import math

        self.batch_size = batch_size
        self._seed = seed
        self.replacement = replacement

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
        rng = random.Random(self._seed)
        conditions = list(self._condition_to_indices.keys())
        for _ in range(self._num_batches):
            cond = rng.choice(conditions)
            pool = self._condition_to_indices[cond]
            if self.replacement:
                yield rng.choices(pool, k=self.batch_size)
            else:
                if len(pool) >= self.batch_size:
                    yield rng.sample(pool, k=self.batch_size)
                else:
                    # repeat-pad to fill batch
                    batch = pool[:]
                    while len(batch) < self.batch_size:
                        batch.extend(
                            rng.sample(
                                pool, k=min(self.batch_size - len(batch), len(pool))
                            )
                        )
                    yield batch[: self.batch_size]
