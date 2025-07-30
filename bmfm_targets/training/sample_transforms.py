"""
Transforms operating on MultiFieldInstance objects.

Used for modifying training logic or augmenting data.
"""

import random
import warnings
from functools import partial, reduce
from operator import itemgetter
from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from transformers.utils import logging

from bmfm_targets.tokenization import MultiFieldInstance
from bmfm_targets.tokenization.resources import (
    get_gene_chromosome_locations,
    get_ortholog_genes,
)

logger = logging.get_logger(__name__)


def encode_expression_as_repeats(
    mfi: MultiFieldInstance,
    chrom_df: pd.DataFrame,
    max_length: int | None = None,
    *args,
    **kwargs,
) -> MultiFieldInstance:
    """
    Convert a single-cell expression profile into a genomic gene sequence
    for training. Sampling reflects abundance; ordering preserves
    spatial genomic structure.

    Based on the input format of
        Universal Cell Embeddings: A Foundation Model for Cell Biology
        Yanay Rosen, Yusuf Roohani, Ayush Agrawal, Leon Samotorčan,
        Tabula Sapiens Consortium, Stephen R. Quake, Jure Leskovec
        bioRxiv 2023.11.28.568918; doi: https://doi.org/10.1101/2023.11.28.568918

    Parameters
    ----------
    mfi : MultiFieldInstance
        Contains 'genes' and 'expressions' from a single cell.
    chrom_df : pd.DataFrame
        Indexed by `gene_symbol`. Must include:
            - 'chromosome': identifier for sorting
            - 'start': genomic start position
    max_length : int
        Desired sequence length. Sampling is with replacement.

    Returns
    -------
    MultiFieldInstance
        Contains reordered 'genes' and corresponding 'expressions'.

    """
    gene_ids = np.array(mfi["genes"])
    expr_values = np.array(mfi["expressions"])
    weights = np.log1p(expr_values)
    if max_length is None:
        max_length = gene_ids.shape[0]
    is_in_chrom_df = np.isin(gene_ids, chrom_df.index)
    filtered_genes = gene_ids[is_in_chrom_df]
    filtered_weights = weights[is_in_chrom_df]
    filtered_exprs = expr_values[is_in_chrom_df]

    sampling_probs = filtered_weights / filtered_weights.sum()
    rng = np.random.default_rng(kwargs.get("seed"))
    sample_idxs = rng.choice(
        len(filtered_genes), size=max_length, replace=True, p=sampling_probs
    )

    sampled_genes = filtered_genes[sample_idxs]
    sampled_exprs = filtered_exprs[sample_idxs]

    gene_positions = chrom_df.loc[sampled_genes]
    chrom_codes = gene_positions["chromosome"].values
    start_coords = gene_positions["start"].values

    shuffled_chroms = np.random.permutation(np.unique(chrom_codes))

    final_genes = []
    final_exprs = []

    for chrom in shuffled_chroms:
        in_chrom = chrom_codes == chrom
        order = np.argsort(start_coords[in_chrom])
        indices = np.where(in_chrom)[0][order]

        final_genes.extend(sampled_genes[indices])
        final_exprs.extend(sampled_exprs[indices])

    return MultiFieldInstance(
        data={"genes": final_genes, "expressions": final_exprs}, metadata=mfi.metadata
    )


def randomize(mfi: MultiFieldInstance, *args, **kwargs) -> MultiFieldInstance:
    indices = np.random.permutation(mfi.seq_length)
    randomized_data = {
        field: np.array(values)[indices].tolist() for field, values in mfi.data.items()
    }
    return MultiFieldInstance(data=randomized_data, metadata=mfi.metadata)


def field_remap(
    mfi: MultiFieldInstance,
    mapping: dict,
    field_to_remap: str = "genes",
    *args,
    **kwargs,
):
    """
    Remaps a field in a MultiFieldInstance object.

    Args:
    ----
        mfi (MultiFieldInstance): mfi
        mapping (dict): A dictionary that maps the original field values to the new values.
        field_to_remap (str): The name of the field to be remapped (default: "genes").

    Returns:
    -------
        MultiFieldInstance: mfi.

    """
    updated_field = [mapping[g] for g in mfi[field_to_remap]]
    updated_data = {}
    for field, vals in mfi.data.items():
        if field == field_to_remap:
            updated_data[field] = updated_field
        else:
            updated_data[field] = vals

    return MultiFieldInstance(
        data=updated_data,
        metadata=mfi.metadata,
    )


def sort_by_field(
    mfi: MultiFieldInstance, field: str, reverse=True, *args, **kwargs
) -> MultiFieldInstance:
    """
    Sort all fields by floating point values of requested field.

    Args:
    ----
        mfi (MultiFieldInstance): mfi
        field (str): name of field
        reverse (bool, optional): reverse sort (True means large to small). Defaults to True.

    Returns:
    -------
        MultiFieldInstance: mfi

    """
    try:
        float_vals = [float(i) for i in mfi.data[field]]
        is_float_field = True
    except ValueError:
        is_float_field = False

    if is_float_field:
        sorted_indices = sorted(
            range(len(mfi.data[field])),
            key=lambda i: float(mfi.data[field][i]),
            reverse=reverse,
        )
    else:
        sorted_indices = sorted(
            range(len(mfi.data[field])),
            key=lambda i: mfi.data[field][i],
            reverse=reverse,
        )
    sorted_data = {}
    for field, values in mfi.data.items():
        sorted_data[field] = [values[i] for i in sorted_indices]
    return MultiFieldInstance(data=sorted_data, metadata=mfi.metadata)


def rda_downsample(
    mfi: MultiFieldInstance,
    max_length: int | None,
    normalized_sum,
    downsample_threshold,
    *args,
    **kwargs,
) -> MultiFieldInstance:
    """
    Implement read-depth aware downsampling of multifield instance.

    Downsample the gene expression values based on the criteria described in
    Hao, Minsheng, et al. "Large-scale foundation model on single-cell transcriptomics."
    Nature Methods (2024): 1-11.

    Note that the norm/log transforms need to be performed in this function
    because they must operate only on the sequence, not the S and T values.

    Args:
    ----
        max_length (int): max length of the sequence
        normalized_sum (int): sum to use for normalizing
        downsample_threshold (int): read count below which not to downsample

    Returns:
    -------
        MultiFieldInstance: RDA transformed MultiFieldInstance with
            additional field "label_expressions" and additional tokens "S" and "T"

    """
    genes = mfi["genes"]
    expressions = mfi["expressions"]

    if max_length is not None:
        genes = genes[: max_length - 2]
        expressions = expressions[: max_length - 2]
    raw_expressions = np.array([float(value) for value in expressions])
    raw_expressions_sum = sum(raw_expressions)
    gamma = (
        0 if raw_expressions_sum < downsample_threshold else np.random.binomial(1, 0.5)
    )
    if gamma == 1:
        beta_distribution = stats.beta(a=2, b=2)
        b = beta_distribution.rvs()
        input_expressions = np.random.binomial(n=raw_expressions.astype(int), p=b)
    else:
        input_expressions = raw_expressions.copy()
    input_expressions_sum = input_expressions.sum()
    if raw_expressions_sum == 0:
        raw_expressions_sum = raw_expressions_sum + 1
    if input_expressions_sum == 0:
        input_expressions = raw_expressions.copy()
        input_expressions_sum = raw_expressions_sum
    raw_expressions = np.log1p((raw_expressions / raw_expressions_sum) * normalized_sum)
    input_expressions = np.log1p(
        (input_expressions / input_expressions_sum) * normalized_sum
    )
    genes = ["[S]", "[T]"] + genes
    ST_list = [np.log1p(input_expressions_sum), np.log1p(raw_expressions_sum)]
    input_expressions = ST_list + input_expressions.tolist()
    raw_expressions = ST_list + raw_expressions.tolist()

    updated_data = {}
    for field, vals in mfi.data.items():
        if field == "genes":
            updated_data[field] = genes
        elif field == "expressions":
            updated_data[field] = input_expressions
            updated_data["label_expressions"] = raw_expressions
        else:
            updated_data[field] = vals

    return MultiFieldInstance(
        data=updated_data,
        metadata=mfi.metadata,
    )


def poisson_downsample(
    mfi: MultiFieldInstance,
    renoise: float = 0.6,
    max_length: int = None,
    log_transform: bool = True,
    *args,
    **kwargs,
) -> MultiFieldInstance:
    """
    Downsamples the gene expression values in a MultiFieldInstance using a Poisson distribution.

    Approach based on Jiang, R., Sun, T., Song, D. et al. Statistics or biology: the zero-inflation
    controversy about scRNA-seq data. Genome Biol 23, 31 (2022) and Kalfon, J., Samaran, J.,
    Peyré, G. et al. scPRINT: pre-training on 50 million cells allows robust gene network
    predictions. Nat Commun 16, 3607 (2025).

    Args:
    ----
        mfi (MultiFieldInstance): The input MultiFieldInstance containing gene and expression data.
        renoise (float, optional): The renoise parameter for the Poisson distribution. Default is 0.6.
        log_transform (bool, optional): Log1p normalize data after downsampling. Default is True.

    Returns:
    -------
        MultiFieldInstance: A new MultiFieldInstance with downsampled expression values.

    """
    genes = mfi["genes"]
    expressions = mfi["expressions"]

    if max_length is not None:
        genes = genes[: max_length - 2]
        expressions = expressions[: max_length - 2]

    n_genes = len(genes)
    raw_expressions = np.array([float(value) for value in expressions])

    if not np.all(np.mod(raw_expressions, 1) == 0):
        raise ValueError("Poisson requires raw expression values.")

    r = renoise * 0.55
    p = stats.poisson(raw_expressions * r)
    rng = np.random.default_rng()
    msk = (rng.random((1, n_genes)) >= r).astype(int)
    input_expressions = (raw_expressions - p.rvs(size=(n_genes,))) * msk
    input_expressions = np.maximum(input_expressions, np.zeros((1, 1), dtype=int))

    raw_expressions_sum = sum(raw_expressions)
    input_expressions_sum = input_expressions.sum()
    genes = ["[S]", "[T]"] + genes
    ST_list = (
        [np.log1p(input_expressions_sum), np.log1p(raw_expressions_sum)]
        if log_transform
        else [input_expressions_sum, raw_expressions_sum]
    )

    if log_transform:
        input_expressions = np.log1p(input_expressions)

    input_expressions = ST_list + [*input_expressions.ravel()]
    raw_expressions = ST_list + [*raw_expressions.ravel()]

    updated_data = {}
    for field, vals in mfi.data.items():
        if field == "genes":
            updated_data[field] = genes
        elif field == "expressions":
            updated_data[field] = input_expressions
            updated_data["label_expressions"] = raw_expressions
        else:
            updated_data[field] = vals

    return MultiFieldInstance(
        data=updated_data,
        metadata=mfi.metadata,
    )


def log_normalize(
    mfi: MultiFieldInstance,
    max_length: int | None,
    normalized_sum=10000,
    *args,
    **kwargs,
) -> MultiFieldInstance:
    """

    Implement log normalization.

    ----
        max_length (int): max length of the sequence
        normalized_sum (int): sum to use for normalizing

    Returns
    -------
        MultiFieldInstance: Log normalized MutifieldInstance

    """
    if max_length is not None:
        for key, val in mfi.data.items():
            mfi.data[key] = val[: max_length - 2]
    expressions = np.array([float(value) for value in mfi.data["expressions"]])
    expressions_sum = sum(expressions) + 1
    expressions = np.log1p((expressions / expressions_sum) * normalized_sum)

    updated_data = {}
    for field, vals in mfi.data.items():
        if field == "expressions":
            updated_data[field] = [*expressions]
        else:
            updated_data[field] = [*vals]

    return MultiFieldInstance(
        data=updated_data,
        metadata=mfi.metadata,
    )


def dropout_random(
    mfi: MultiFieldInstance, dropout_ratio: float, *args, **kwargs
) -> MultiFieldInstance:
    """
    Dropout elements at random across all fields.

    Args:
    ----
        mfi (MultiFieldInstance): mfi
        dropout_ratio (float): probability of being dropped out

    Returns:
    -------
        MultiFieldInstance: mfi

    """
    assert 0 <= dropout_ratio < 1
    sequence_len = len(next(iter(mfi.data.values())))
    dropout_indices = [
        i for i in range(sequence_len) if dropout_ratio > random.random()
    ]
    data_after_dropout = {}
    for field, values in mfi.data.items():
        data_after_dropout[field] = [
            elem for i, elem in enumerate(values) if i not in dropout_indices
        ]
    return MultiFieldInstance(data=data_after_dropout, metadata=mfi.metadata)


def dropout_chunk_in_range(
    mfi: MultiFieldInstance,
    chunk_size: int,
    drop_range: tuple | None = None,
    *args,
    **kwargs,
) -> MultiFieldInstance:
    """
    Dropout contiguous block from MultiFieldInstance.

    Args:
    ----
        mfi (MultiFieldInstance): mfi
        chunk_size (int): size of chunk to drop
        drop_range (tuple | None, optional): range within which to drop. Defaults to None.

    Returns:
    -------
        MultiFieldInstance: mfi

    """
    if drop_range is None:
        drop_range = (0, mfi.seq_length - 1)

    range_len = drop_range[1] - drop_range[0] + 1
    if mfi.seq_length < range_len:
        drop_range = (drop_range[0], drop_range[0] + mfi.seq_length - 1)
        range_len = drop_range[1] - drop_range[0] + 1

    if chunk_size > range_len:
        warnings.warn(
            f"Requested to drop chunk of size {chunk_size} in range of length {range_len}. Ignoring."
        )
        return mfi

    start_index = random.randint(0, range_len - chunk_size)
    stop_index = start_index + chunk_size
    data_after_dropout = {}
    for field, values in mfi.data.items():
        data_after_dropout[field] = values[:start_index] + values[stop_index:]
    return MultiFieldInstance(data=data_after_dropout, metadata=mfi.metadata)


def rda_align(
    mfi: MultiFieldInstance,
    target_read_resolution: int,
    normalized_sum: int,
    max_length: int | None = None,
    *args,
    **kwargs,
) -> MultiFieldInstance:
    """
    Read-Depth align a sample for inference.

    Insert target read resolution in the [T] position and input read resolution in the [S]
    position, for aligning read depth in a model that was trained with rda_downsampling.

    Args:
    ----
        mfi (MultiFieldInstance): MultiFieldInstance with raw integer read counts
        target_read_resolution (int): target read resolution - sample will be embedded
          as if it were being upsampled to this resolution
        normalized_sum (int): sum for normalization
        max_length (int | None, optional): max length so that the sums of reads are
          calculated correctly. Defaults to None.

    Returns:
    -------
        MultiFieldInstance: _description_

    """
    genes = mfi["genes"]
    expressions = mfi["expressions"]

    if max_length is not None:
        genes = genes[: max_length - 2]
        expressions = expressions[: max_length - 2]

    raw_expressions = np.array([float(value) for value in expressions])
    raw_expressions_sum = sum(raw_expressions)
    if raw_expressions_sum > 0:
        normed_expressions = np.log1p(
            (raw_expressions / raw_expressions_sum) * normalized_sum
        )
    else:
        normed_expressions = raw_expressions

    genes = ["[S]", "[T]"] + genes
    ST_list = [np.log1p(raw_expressions_sum), np.log1p(target_read_resolution)]
    rda_aligned_expressions = ST_list + normed_expressions.tolist()

    updated_data = {}
    for field, vals in mfi.data.items():
        if field == "genes":
            updated_data[field] = genes
        elif field == "expressions":
            updated_data[field] = rda_aligned_expressions
        else:
            updated_data[field] = vals

    return MultiFieldInstance(
        data=updated_data,
        metadata=mfi.metadata,
    )


def pad_zero_expressed_genes(
    mfi: MultiFieldInstance,
    pad_zero_expression_strategy: dict,
    max_length: int,
    *args,
    **kwargs,
):
    """
    Reorders and truncates genes based on their expression levels according to the specified padding strategy.

    Parameters
    ----------
        mfi (MultiFieldInstance): The input instance containing gene data.
        pad_zero_expression_strategy (dict): The strategy for handling zero-expression genes.
            required key "strategy" can have values:
            - 'batch_wise': Prioritizes genes expressed in the batch.
            - 'random': Randomly selects zero-expression genes if needed.
            optional keys:
            - interleave_zero_ratio (float): Interleave a fixed ratio of zeros
              (0 means first include all nonzero values, 1 means put all zeros first)
        max_length (int): Maximum number of genes to retain.
        *args: Unused positional arguments.
        **kwargs: Additional keyword arguments.
            - expressed_genes_in_batch (set): Required for 'batch_wise' strategy.


    Returns
    -------
        MultiFieldInstance: A new instance with reordered and truncated gene data.

    """
    final_kwargs = {
        **kwargs,
        **{k: v for k, v in pad_zero_expression_strategy.items() if k != "strategy"},
    }
    if pad_zero_expression_strategy["strategy"] == "batch_wise":
        return _batchwise_pad_zero_expressed_genes(mfi, max_length, **final_kwargs)

    elif pad_zero_expression_strategy["strategy"] == "random":
        return _random_pad_zero_expressed_genes(mfi, max_length, **final_kwargs)
    raise ValueError(
        f"Unrecognized zero padding strategy = {pad_zero_expression_strategy}"
    )


def _random_pad_zero_expressed_genes(mfi: MultiFieldInstance, max_length: int):
    data_to_keep, zero_indices, keep_indices = {}, [], []

    for i, expression in enumerate(mfi.data["expressions"]):
        (keep_indices if expression > 0.0 else zero_indices).append(i)

    if len(keep_indices) < max_length:
        needed = max_length - len(keep_indices)
        keep_indices.extend(random.sample(zero_indices, min(len(zero_indices), needed)))

    data_to_keep = {
        field: list(itemgetter(*keep_indices)(values))
        for field, values in mfi.data.items()
    }
    return MultiFieldInstance(data=data_to_keep, metadata=mfi.metadata)


def _batchwise_pad_zero_expressed_genes(
    mfi: MultiFieldInstance,
    max_length: int,
    expressed_genes_in_batch: set,
    interleave_zero_ratio: float | None = None,
):
    sample_nz, batch_nz, non_expressed_indices = [], [], []

    for i, (gene, expression) in enumerate(
        zip(mfi.data["genes"], mfi.data["expressions"])
    ):
        if expression > 0.0:
            sample_nz.append(i)
        elif gene in expressed_genes_in_batch:
            batch_nz.append(i)
        else:
            non_expressed_indices.append(i)

    if interleave_zero_ratio is not None:
        combined_expressed = _interleave_with_ratio(
            sample_nz, batch_nz, interleave_zero_ratio
        )
    else:
        combined_expressed = sample_nz + batch_nz

    ordered_indices = (combined_expressed + non_expressed_indices)[:max_length]
    updated_data = {
        field: list(itemgetter(*ordered_indices)(values))
        for field, values in mfi.data.items()
    }
    return MultiFieldInstance(data=updated_data, metadata=mfi.metadata)


def _interleave_with_ratio(a: list, b: list, p: float) -> list:
    """
    Interleave elements from two lists according to a specified ratio.

    Parameters
    ----------
    a : list
        First input list.
    b : list
        Second input list.
    p : float
        Ratio parameter between 0 and 1, representing the proportion of elements
        to take from list 'a' in each interleaving cycle.

    Returns
    -------
    list
        A new list containing interleaved elements from 'a' and 'b' according to
        the ratio 'p', followed by any remaining elements from both lists.

    Raises
    ------
    ValueError
        If 'p' is not in the range [0, 1].

    """
    if not (0 <= p <= 1):
        raise ValueError(f"p must be in [0,1], got {p}")

    if p == 0:
        return b + a
    if p == 1:
        return a + b
    if not a or not b:
        return a + b

    k = max(1, round(1 / p)) if p > 0 else 1
    a_step = max(1, round(p * k))
    b_step = k - a_step
    min_len = min(len(a) // a_step, len(b) // b_step)

    if min_len == 0:
        return a + b

    sample_part = np.array(a[: min_len * a_step]).reshape(min_len, a_step)
    batch_part = np.array(b[: min_len * b_step]).reshape(min_len, b_step)
    interleaved = np.hstack([sample_part, batch_part]).flatten().tolist()

    return interleaved + a[min_len * a_step :] + b[min_len * b_step :]


def downcast_numeric_fields(
    mfi: MultiFieldInstance,
    fields_to_downcast: list[str],
    casting_strategy: Literal["round"] | Literal["ceil"] | Literal["floor"] = "round",
    *args,
    **kwargs,
):
    """
    Downcast floating point fields to integers for tokenization.

    Args:
    ----
        mfi (MultiFieldInstance): input mfi
        fields_to_downcast (list[str]): list of fields to downcast
        casting_strategy ("round","ceil","floor"):
            Whether to downcast via round, floor or ceil. Defaults to "round".

    Returns:
    -------
        _type_: mfi with integer fields in place of floats

    """
    np_op = {"round": np.round, "floor": np.floor, "ceil": np.ceil}

    data = {
        k: (
            np_op[casting_strategy](v).astype(int).tolist()
            if k in fields_to_downcast
            else v
        )
        for k, v in mfi
    }
    return MultiFieldInstance(data=data, metadata=mfi.metadata)


def compose_transforms(
    sequence_order=None,
    log_normalize_transform=False,
    rda_transform=None,
    pad_zero_expression_strategy=None,
    max_length=None,
    sequence_dropout_factor=None,
    fields_to_downcast=None,
    map_orthologs=None,
    renoise=None,
):
    transforms = []
    if sequence_order == "random":
        transforms.append(randomize)
    elif sequence_order == "sorted":
        transforms.append(partial(sort_by_field, field="expressions"))
    elif sequence_order == "chromosomal_uce":
        chrom_df = get_gene_chromosome_locations(species="human")
        transforms.append(
            partial(
                encode_expression_as_repeats, chrom_df=chrom_df, max_length=max_length
            )
        )
    if sequence_dropout_factor is not None:
        if sequence_dropout_factor > 1:
            transforms.append(
                partial(
                    dropout_chunk_in_range,
                    chunk_size=sequence_dropout_factor,
                    drop_range=(0, max_length - 1),
                )
            )
        else:
            transforms.append(
                partial(
                    dropout_random,
                    dropout_ratio=sequence_dropout_factor,
                )
            )

    if pad_zero_expression_strategy is not None:
        transforms.append(
            partial(
                pad_zero_expressed_genes,
                pad_zero_expression_strategy=pad_zero_expression_strategy,
                max_length=max_length,
            )
        )

    if log_normalize_transform:
        transforms.append(partial(log_normalize, max_length=max_length))

    if rda_transform == "downsample":
        transforms.append(
            partial(
                rda_downsample,
                max_length=max_length,
                downsample_threshold=1000,
                normalized_sum=10000,
            )
        )
    elif rda_transform == "poisson_downsample":
        transforms.append(
            partial(
                poisson_downsample,
                renoise=renoise,
                max_length=max_length,
                log_normalize_transform=True,
            )
        )
    elif rda_transform == "equal":
        transforms.append(
            partial(
                rda_downsample,
                max_length=max_length,
                downsample_threshold=torch.inf,
                normalized_sum=10000,
            )
        )
    elif isinstance(rda_transform, int) and not isinstance(rda_transform, bool):
        transforms.append(
            partial(
                rda_align,
                target_read_resolution=rda_transform,
                max_length=max_length,
                normalized_sum=10000,
            )
        )
    if fields_to_downcast:
        transforms.append(
            partial(
                downcast_numeric_fields,
                fields_to_downcast=fields_to_downcast,
            )
        )
    if map_orthologs == "mouse_to_human_orthologs":
        mapping = get_ortholog_genes(
            return_mapping=True,
            from_species="mmusculus",
            to_species="hsapiens",
            id_type="gene_name",
        )
        mapping.update({"[S]": "[S]", "[T]": "[T]"})
        transforms.append(
            partial(
                field_remap,
                field_to_remap="genes",
                mapping=mapping,
            )
        )
    elif map_orthologs == "human_to_mouse_orthologs":
        mapping = get_ortholog_genes(
            return_mapping=True,
            from_species="hsapiens",
            to_species="mmusculus",
            id_type="gene_name",
        )
        mapping.update({"[S]": "[S]", "[T]": "[T]"})
        transforms.append(
            partial(
                field_remap,
                field_to_remap="genes",
                mapping=mapping,
            )
        )

    return transforms


def get_genes_expressed_in_batch(examples):
    return {
        gene
        for mfi in examples
        for gene, expression in zip(mfi.data["genes"], mfi.data["expressions"])
        if expression != 0.0
    }


def transform_inputs(
    examples,
    fields,
    sequence_order,
    log_normalize_transform,
    rda_transform,
    pad_zero_expression_strategy,
    max_length,
    sequence_dropout_factor,
    map_orthologs,
    renoise,
):
    fields_to_downcast = [
        f.field_name
        for f in fields
        if "expressions" in f.field_name
        and f.tokenization_strategy != "continuous_value_encoder"
        and f.is_input
    ]
    transforms = compose_transforms(
        sequence_order=sequence_order,
        log_normalize_transform=log_normalize_transform,
        rda_transform=rda_transform,
        pad_zero_expression_strategy=pad_zero_expression_strategy,
        max_length=max_length,
        sequence_dropout_factor=sequence_dropout_factor,
        fields_to_downcast=fields_to_downcast,
        map_orthologs=map_orthologs,
        renoise=renoise,
    )
    if "perturbations" in [i.field_name for i in fields]:
        transforms.append(partial(sort_by_field, field="perturbations"))
    if len(transforms) > 0:
        combined_func = reduce(
            lambda f, g: lambda x, *a, **k: g(f(x, *a, **k), *a, **k),
            transforms,
        )
        if (
            pad_zero_expression_strategy is not None
            and pad_zero_expression_strategy["strategy"] == "batch_wise"
        ):
            expressed_genes_in_batch = get_genes_expressed_in_batch(examples)
        else:
            expressed_genes_in_batch = {}
        return [
            combined_func(x, expressed_genes_in_batch=expressed_genes_in_batch)
            for x in examples
        ]
    return examples
