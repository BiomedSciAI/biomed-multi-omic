import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import read_h5ad
from scipy.sparse import rand

from bmfm_targets.datasets import zheng68k
from bmfm_targets.datasets.base_rna_dataset import (
    BaseRNAExpressionDataset,
    SparseDataset,
    in_silico_perturbation_wrapper,
)
from bmfm_targets.datasets.zheng68k import Zheng68kDataModule
from bmfm_targets.tokenization import load_tokenizer
from bmfm_targets.tokenization.resources import get_hgnc_df, get_protein_coding_genes
from bmfm_targets.training.data_module import DataModule
from bmfm_targets.training.sample_transforms import sort_by_field

from .helpers import Zheng68kPaths


def test_backed_csr_and_in_memory_csr_reads_are_the_same():
    data_path = Zheng68kPaths.root / "h5ad" / "zheng68k.h5ad"
    ad = read_h5ad(data_path)
    ad_backed = read_h5ad(data_path, backed="r")
    all_genes = ad_backed.var_names
    for idx in range(100):
        g, e = BaseRNAExpressionDataset.read_expressions_and_genes_from_csr(
            ad.X, all_genes, idx
        )
        (
            g_backed,
            e_backed,
        ) = BaseRNAExpressionDataset.read_expressions_and_genes_from_csr(
            ad_backed.X, all_genes, idx
        )
        np.testing.assert_array_equal(g, g_backed)
        np.testing.assert_array_equal(e, e_backed)


def test_all_zeros_exposed_correctly(pl_data_module_zheng68k_seq_cls):
    ds = zheng68k.Zheng68kDataset(
        processed_data_source=pl_data_module_zheng68k_seq_cls.processed_data_file,
        expose_zeros="all",
    )
    mfi = ds[0]

    data_row = ds.processed_data[0].to_df().squeeze()
    original_genes, original_expressions = [*zip(*data_row.sort_index().items())]

    sorted_mfi = sort_by_field(mfi, "genes", reverse=False)
    assert sorted_mfi["genes"] == list(original_genes)
    assert sorted_mfi["expressions"] == list(original_expressions)


def test_ordering_mfi_in_presence_of_perturbation(pl_data_module_zheng68k_seq_cls):
    ds = zheng68k.Zheng68kDataset(
        processed_data_source=pl_data_module_zheng68k_seq_cls.processed_data_file,
        output_wrapper=partial(
            in_silico_perturbation_wrapper,
            perturbations={"RPL11": "max*2+6"},
        ),
    )
    mfi = ds[0]

    sorted_mfi = sort_by_field(mfi, "perturbations")
    assert sorted_mfi["genes"][0] == "RPL11"
    assert sorted_mfi["expressions"][0] == 22


def test_get_zero_expression_genes(pl_data_module_zheng68k_seq_cls):
    ds = zheng68k.Zheng68kDataset(
        processed_data_source=pl_data_module_zheng68k_seq_cls.processed_data_file,
        expose_zeros="all",
    )
    data_row = ds.processed_data[0].to_df().squeeze()
    assert not isinstance(ds.binned_data, SparseDataset)
    genes, expressions = ds.get_genes_and_expressions(0)
    zero_genes = [g for idx, g in enumerate(genes) if expressions[idx] == 0]
    assert zero_genes == [*data_row[data_row == 0].index]


def test_expose_zeros_other_than_all_not_implemented(pl_data_module_zheng68k_seq_cls):
    with pytest.raises(NotImplementedError, match="Unsupported option"):
        ds = zheng68k.Zheng68kDataset(
            processed_data_source=pl_data_module_zheng68k_seq_cls.processed_data_file,
            expose_zeros="topk",
        )


def test_expose_zeros_none_has_no_zeros(pl_data_module_zheng68k_seq_cls):
    ds = zheng68k.Zheng68kDataset(
        processed_data_source=pl_data_module_zheng68k_seq_cls.processed_data_file,
        expose_zeros=None,
    )
    mfi = ds[0]
    assert "0" not in mfi["expressions"]


def test_sort_data():
    adata, genes_ordered = create_random_dataset_with_gene_order_col()

    ds_sorted = BaseRNAExpressionDataset(
        processed_data_source=adata.copy(),
        sort_genes_var="gene_order",
        expose_zeros="all",
    )
    ds_unsorted = BaseRNAExpressionDataset(
        processed_data_source=adata.copy(), expose_zeros="all"
    )

    unsorted_mfi = ds_unsorted[0]
    assert unsorted_mfi["genes"] == [*adata.var.index]
    sorted_mfi = ds_sorted[0]
    assert sorted_mfi["genes"] == genes_ordered


def test_get_zero_expression_gene_in_backed_mode(pl_data_module_zheng68k_seq_cls):
    ds = zheng68k.Zheng68kDataset(
        processed_data_source=pl_data_module_zheng68k_seq_cls.processed_data_file,
        expose_zeros="all",
        split=None,
        backed="r",
    )
    data_row = ds.processed_data[0].to_df().squeeze()
    assert isinstance(ds.binned_data, SparseDataset)
    genes, expressions = ds.get_genes_and_expressions(0)
    zero_genes = [g for idx, g in enumerate(genes) if expressions[idx] == 0]
    assert zero_genes == [*data_row[data_row == 0].index]


def create_random_dataset_with_gene_order_col():
    n_obs = 10
    n_vars = 10
    X = rand(n_obs, n_vars, density=0.2, format="csr")

    adata = sc.AnnData(X)
    adata.obs_names = [f"cell{i}" for i in range(n_obs)]
    adata.var_names = [f"gene{j}" for j in range(n_vars)]

    adata.var["gene_order"] = [8, 7, 5, 1, 4, 3, 0, 9, 2, 6]

    genes_ordered = [
        gene
        for gene, _ in sorted(
            zip(adata.var_names, adata.var["gene_order"]),
            key=lambda x: x[1],
            reverse=True,
        )
    ]

    return adata, genes_ordered


@pytest.mark.skip(reason="sps integration fails, requires additional review")
def test_data_module_setup_predict_gives_all_splits(
    pl_data_module_zheng68k_seq_cls: Zheng68kDataModule,
):
    data_module = Zheng68kDataModule(
        dataset_kwargs={
            "processed_data_source": pl_data_module_zheng68k_seq_cls.processed_data_file,
            # "filter_query": "split_stratified_celltype != 'test'"
        },
        tokenizer=pl_data_module_zheng68k_seq_cls.tokenizer,
        fields=pl_data_module_zheng68k_seq_cls.fields,
        label_columns=pl_data_module_zheng68k_seq_cls.label_columns,
        transform_datasets=False,
        collation_strategy="sequence_classification",
        batch_size=1,
        max_length=32,
    )
    data_module.setup("predict")
    value_counts = (
        data_module.predict_dataset.processed_data.obs["split_stratified_celltype"]
        .value_counts()
        .to_dict()
    )
    assert sorted(value_counts.keys()) == ["dev", "test", "train"]


@pytest.mark.skip(reason="sps integration fails, requires additional review")
def test_data_module_setup_test_gives_test_only(
    pl_data_module_zheng68k_seq_cls: Zheng68kDataModule,
):
    data_module = Zheng68kDataModule(
        dataset_kwargs={
            "processed_data_source": pl_data_module_zheng68k_seq_cls.processed_data_file,
        },
        tokenizer=pl_data_module_zheng68k_seq_cls.tokenizer,
        fields=pl_data_module_zheng68k_seq_cls.fields,
        label_columns=pl_data_module_zheng68k_seq_cls.label_columns,
        transform_datasets=False,
        collation_strategy="sequence_classification",
        batch_size=1,
        max_length=32,
    )
    data_module.setup("test")
    value_counts = (
        data_module.test_dataset.processed_data.obs["split_stratified_celltype"]
        .value_counts()
        .to_dict()
    )
    assert sorted(value_counts.keys()) == ["test"]


def test_generic_base_rna_dataset_with_label_columns(pl_data_module_zheng68k_seq_cls):
    with tempfile.TemporaryDirectory() as d:
        ds = BaseRNAExpressionDataset(
            processed_data_source=pl_data_module_zheng68k_seq_cls.dataset_kwargs[
                "processed_data_source"
            ],
            split="train",
            label_columns=["celltype"],
            stratifying_label=pl_data_module_zheng68k_seq_cls.stratifying_label,
            label_dict_path=Path(d) / "label_dict.json",
        )
        assert ds.label_dict is not None


def test_generic_base_rna_dataset_limiting_genes(pl_data_module_zheng68k_seq_cls):
    protein_coding_genes = get_hgnc_df(
        'locus_group == "protein-coding gene"'
    ).symbol.to_list()
    assert sorted(protein_coding_genes) == sorted(get_protein_coding_genes())

    ds = BaseRNAExpressionDataset(
        processed_data_source=pl_data_module_zheng68k_seq_cls.dataset_kwargs[
            "processed_data_source"
        ],
        split="train",
        stratifying_label=pl_data_module_zheng68k_seq_cls.stratifying_label,
        limit_genes=protein_coding_genes,
    )
    ds_all_genes = BaseRNAExpressionDataset(
        processed_data_source=pl_data_module_zheng68k_seq_cls.dataset_kwargs[
            "processed_data_source"
        ],
        split="train",
        stratifying_label=pl_data_module_zheng68k_seq_cls.stratifying_label,
    )
    mfi = ds[0]
    assert all(g in protein_coding_genes for g in mfi["genes"])

    mfi_all_genes = ds_all_genes[0]
    s1 = pd.Series(dict(zip(mfi_all_genes["genes"], mfi_all_genes["expressions"])))
    s2 = pd.Series(dict(zip(mfi["genes"], mfi["expressions"])))
    # there are more genes without limiting
    assert len(s1) > len(s2)
    # where the genes overlap the expressions are equal
    assert pd.concat([s1, s2], axis=1).dropna().diff(axis=1).iloc[:, 1].sum() == 0


def test_generic_base_rna_dataset_no_label_columns(pl_data_module_zheng68k_seq_cls):
    processed_data_source = pl_data_module_zheng68k_seq_cls.dataset_kwargs[
        "processed_data_source"
    ]
    ds = BaseRNAExpressionDataset(
        processed_data_source=processed_data_source, split=None
    )
    assert not ds.labels_requested
    assert ds.label_dict is None


def test_data_module_init_with_generic_dataset_seq_cls_transform_true(
    gene2vec_fields, zheng68k_label_columns
):
    with tempfile.TemporaryDirectory() as d:
        dm = DataModule(
            tokenizer=load_tokenizer(),
            fields=gene2vec_fields,
            label_columns=zheng68k_label_columns,
            collation_strategy="sequence_classification",
            dataset_kwargs={
                "processed_data_source": Path(d) / "processed.h5ad",
                "label_dict_path": Path(d) / "label_dict.json",
            },
            transform_kwargs={
                "source_h5ad_file_name": Zheng68kPaths.root / "h5ad" / "zheng68k.h5ad",
                "processed_h5ad_file_name": Path(d) / "processed.h5ad",
                "stratifying_label": "celltype",
            },
        )
        dm.prepare_data()
        dm.setup("fit")
        for batch in dm.train_dataloader():
            assert "celltype" in batch["labels"]
            break


def test_data_module_init_with_generic_dataset_seq_cls_transform_false(
    pl_data_module_zheng68k_seq_cls,
):
    fields = pl_data_module_zheng68k_seq_cls.fields
    label_columns = pl_data_module_zheng68k_seq_cls.label_columns
    tokenizer = pl_data_module_zheng68k_seq_cls.tokenizer
    processed_path = pl_data_module_zheng68k_seq_cls.dataset_kwargs[
        "processed_data_source"
    ]
    with tempfile.TemporaryDirectory() as d:
        dm = DataModule(
            tokenizer=tokenizer,
            fields=fields,
            label_columns=label_columns,
            collation_strategy="sequence_classification",
            dataset_kwargs={
                "processed_data_source": processed_path,
            },
            transform_datasets=False,
        )
        dm.prepare_data()
        dm.setup("fit")
        for batch in dm.train_dataloader():
            assert "celltype" in batch["labels"]
            break


def test_data_module_init_with_generic_dataset_mlm(gene2vec_fields):
    with tempfile.TemporaryDirectory() as d:
        dm = DataModule(
            tokenizer=load_tokenizer(),
            fields=gene2vec_fields,
            collation_strategy="language_modeling",
            mlm=True,
            dataset_kwargs={
                "processed_data_source": Path(d) / "processed.h5ad",
            },
            transform_kwargs={
                "source_h5ad_file_name": Zheng68kPaths.root / "h5ad" / "zheng68k.h5ad",
                "processed_h5ad_file_name": Path(d) / "processed.h5ad",
                "stratifying_label": None,
            },
        )
        dm.prepare_data()
        dm.setup("fit")
        for batch in dm.train_dataloader():
            assert batch["input_ids"].shape[1] == 2  # has two fields of input
            break


def test_perturbation_get_item_includes_perturbed_genes(
    pl_data_module_adamson_weissman_seq_labeling,
):
    ds = pl_data_module_adamson_weissman_seq_labeling.train_dataset
    all_genes = ds.all_genes
    for i in range(len(ds)):
        mfi = ds[i]
        perturbed_genes = mfi.metadata["perturbed_genes"]
        assert isinstance(perturbed_genes, str)
        assert all(isinstance(g, str) for g in perturbed_genes.split("_"))
        assert all(g in mfi["genes"] for g in perturbed_genes.split("_"))
        assert len(perturbed_genes.split("_")) > 0
    for batch in pl_data_module_adamson_weissman_seq_labeling.train_dataloader():
        perturbed_genes = batch["perturbed_genes"]
        assert all(isinstance(pg, str) for pg in perturbed_genes)
        assert all(pg.split("_")[0] in all_genes for pg in perturbed_genes)


def test_perturbation_get_item_includes_delta_from_baseline(
    pl_data_module_adamson_weissman_seq_labeling,
):
    ds = pl_data_module_adamson_weissman_seq_labeling.train_dataset
    for i in range(len(ds)):
        mfi = ds[i]
        diff = np.array(mfi["label_expressions"]) - np.array(
            mfi["delta_baseline_expressions"]
        )
        expected = ds.avg_pert_expressions.loc[mfi["genes"]].to_numpy()
        np.testing.assert_allclose(diff, expected, atol=1.0e-6)


def test_perturbation_dataset_respects_expose_zeros(
    pl_data_module_adamson_weissman_seq_labeling,
):
    dl = pl_data_module_adamson_weissman_seq_labeling
    from bmfm_targets.datasets.base_perturbation_dataset import BasePerturbationDataset

    ds = BasePerturbationDataset(expose_zeros="all", **dl.dataset_kwargs)
    all_genes = ds.all_genes
    for i in range(len(ds)):
        mfi = ds[i]
        assert len(mfi["genes"]) == len(all_genes)
        assert any(i == 0 for i in mfi["expressions"])

    ds = BasePerturbationDataset(expose_zeros="no_zeros", **dl.dataset_kwargs)
    all_genes = ds.all_genes
    for i in range(len(ds)):
        mfi = ds[i]
        zero_ratio = (
            sum(
                e == 0
                for e, p in zip(mfi["expressions"], mfi["perturbations"])
                if int(p) == 0
            )
            / mfi.seq_length
        )
        assert zero_ratio == 0
