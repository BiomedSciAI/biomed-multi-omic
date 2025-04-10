import scanpy as sc

from bmfm_targets.datasets.zheng68k import Zheng68kDataModule, Zheng68kDataset
from bmfm_targets.evaluation import evaluate_clusters, generate_clusters
from bmfm_targets.tokenization import get_gene2vec_tokenizer

from ..helpers import (
    MockTestDataPaths,
    check_h5ad_file_is_csr,
    check_h5ad_file_structure,
    check_splits,
    check_unk_levels_for_dm,
    ensure_expected_indexes_and_labels_present,
)


def test_zhenk68k_csr_matrix_load_type(mock_data_dataset_kwargs_after_transform):
    ds_train = Zheng68kDataset(**mock_data_dataset_kwargs_after_transform, split="train")
    check_h5ad_file_is_csr(ds_train.processed_data)


def test_zheng68k_h5ad_structure():
    h5ad_path = MockTestDataPaths.root / "h5ad" / "mock_test_data.h5ad"
    check_h5ad_file_structure(h5ad_path)


def test_loaded_samples_are_valid(mock_data_dataset_kwargs_after_transform):
    ensure_expected_indexes_and_labels_present(
        mock_data_dataset_kwargs_after_transform,
        label_column_names=["celltype", "cell_type_ontology_term_id"],
        dataset_factory=Zheng68kDataset,
    )


def test_zhenk68k_mapped_cellxgene_types_exists(
    mock_data_dataset_kwargs_after_transform_without_labels,
):
    ds = Zheng68kDataset(
        **mock_data_dataset_kwargs_after_transform_without_labels,
        split="train",
        label_columns=["celltype", "cell_type_ontology_term_id"],
    )
    assert len(ds[0].metadata) == 3
    assert ds[0].metadata["cell_type_ontology_term_id"].startswith("CL:")


def test_zheng68k_data_module_sequence_prediction(
    mock_data_dataset_kwargs_after_transform, gene2vec_fields, mock_data_label_columns
):
    dm = Zheng68kDataModule(
        dataset_kwargs=mock_data_dataset_kwargs_after_transform,
        tokenizer=get_gene2vec_tokenizer(),
        fields=gene2vec_fields,
        label_columns=mock_data_label_columns,
        transform_datasets=False,
        collation_strategy="sequence_classification",
    )
    dm.prepare_data()
    dm.setup("fit")
    label = dm.train_dataset.label_columns[0]
    for batch in dm.train_dataloader():
        assert {*batch["labels"][label].numpy()}.issubset(dm.label_dict[label].values())
        break
    for batch in dm.val_dataloader():
        assert {*batch["labels"][label].numpy()}.issubset(dm.label_dict[label].values())
        break


def test_zheng68k_data_module_mlm(
    mock_data_dataset_kwargs_after_transform_without_labels, gene2vec_fields
):
    dm = Zheng68kDataModule(
        dataset_kwargs=mock_data_dataset_kwargs_after_transform_without_labels,
        tokenizer=get_gene2vec_tokenizer(),
        fields=gene2vec_fields,
        transform_datasets=False,
        mlm=True,
        collation_strategy="language_modeling",
    )
    dm.prepare_data()
    dm.setup("fit")
    for batch in dm.train_dataloader():
        assert ((batch["labels"]["expressions"].numpy() == -100).mean(axis=0) > 0).all()
        break
    for batch in dm.val_dataloader():
        assert ((batch["labels"]["expressions"].numpy() == -100).mean(axis=0) > 0).all()
        break


def test_zheng68k_unk_levels(
    mock_data_dataset_kwargs_after_transform_without_labels, gene2vec_fields
):
    dm = Zheng68kDataModule(
        dataset_kwargs=mock_data_dataset_kwargs_after_transform_without_labels,
        tokenizer=get_gene2vec_tokenizer(),
        fields=gene2vec_fields,
        transform_datasets=False,
        mlm=True,
        collation_strategy="language_modeling",
        limit_genes=None,
    )
    dm.prepare_data()
    dm.setup("fit")
    check_unk_levels_for_dm(dm)


def test_splits():
    check_splits(
        MockTestDataPaths.root / "h5ad" / "mock_test_data.h5ad",
        split_weights={"train": 0.8, "dev": 0.1, "test": 0.1},
        test_needle="CAAGTTCTATCGAC-2",
        balancing_label="celltype",
    )


def test_clustering_of_raw_expressions():
    h5ad_path = MockTestDataPaths.root / "h5ad" / "mock_test_data.h5ad"
    adata = sc.read_h5ad(h5ad_path)
    adata.X = adata.X.astype("float64")

    clusters = generate_clusters(adata, clustering_method="kmeans")
    eval_results = evaluate_clusters(
        clusters, clustering_method="kmeans", label="celltype"
    )
    assert round(eval_results["ARI"], 2) > 0.08
    assert round(eval_results["ARI"], 2) < 0.2

    clusters = generate_clusters(adata, clustering_method="louvain")
    eval_results = evaluate_clusters(
        clusters, clustering_method="louvain", label="celltype"
    )
    assert round(eval_results["ARI"], 2) > 0.05
    assert round(eval_results["ARI"], 2) < 0.1
