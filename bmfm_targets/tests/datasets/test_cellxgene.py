import pytest

try:
    # anndata >= 0.11
    from anndata.abc import CSRDataset as SparseDataset
except ImportError:
    # anndata >= 0.10
    from anndata.experimental import CSRDataset as SparseDataset
from scipy.sparse import csr_matrix

from bmfm_targets.datasets.cellxgene import (
    CellXGeneDataModule,
    CellXGeneDataset,
)
from bmfm_targets.tests.helpers import (
    CellXGenePaths,
    check_h5ad_file_is_csr,
    check_h5ad_file_structure,
    check_splits,
    check_unk_levels_for_dm,
    ensure_expected_indexes_and_labels_present,
    test_pre_transforms,
)
from bmfm_targets.tokenization import get_gene2vec_tokenizer


@pytest.fixture(scope="module")
def cellxgene_dm_transform_true(gene2vec_fields, cellxgene_label_columns):
    ds_kwargs = {
        "label_dict_path": CellXGenePaths.label_dict_path,
    }
    transform_kwargs = {
        "transforms": test_pre_transforms,
        "split_weights": {"train": 0.4, "dev": 0.3, "test": 0.3},
    }
    dm = CellXGeneDataModule(
        dataset_kwargs=ds_kwargs,
        label_columns=cellxgene_label_columns,
        transform_kwargs=transform_kwargs,
        data_dir=CellXGenePaths.root,
        tokenizer=get_gene2vec_tokenizer(),
        fields=gene2vec_fields,
        transform_datasets=True,
        mlm=False,
        collation_strategy="sequence_classification",
    )
    dm.prepare_data()
    dm.setup()
    return dm


@pytest.fixture()
def already_transformed_kwargs(cellxgene_dm_transform_true):
    dataset_kwargs = cellxgene_dm_transform_true.dataset_kwargs
    dataset_kwargs[
        "processed_data_source"
    ] = cellxgene_dm_transform_true.processed_data_file
    return dataset_kwargs


def test_cellxgene_csr_matrix_load_type(already_transformed_kwargs):
    ds_train = CellXGeneDataset(**already_transformed_kwargs, split="train")
    check_h5ad_file_is_csr(ds_train.processed_data)
    ds_test = CellXGeneDataset(**already_transformed_kwargs, split="test")
    check_h5ad_file_is_csr(ds_test.processed_data)
    ds_dev = CellXGeneDataset(**already_transformed_kwargs, split="dev")
    check_h5ad_file_is_csr(ds_dev.processed_data)


def test_cellxgene_h5ad_structure():
    h5ad_path = CellXGenePaths.root / "h5ad" / "cellxgene.h5ad"
    check_h5ad_file_structure(h5ad_path)


def test_cellxgene_unk_levels(already_transformed_kwargs, gene2vec_fields):
    dm = CellXGeneDataModule(
        dataset_kwargs=already_transformed_kwargs,
        tokenizer=get_gene2vec_tokenizer(),
        fields=gene2vec_fields,
        transform_datasets=False,
        mlm=False,
        collation_strategy="language_modeling",
        limit_genes=None,
    )

    dm.prepare_data()
    dm.setup()
    check_unk_levels_for_dm(dm, unk_frac_threshold=0.15)


def test_loaded_samples_are_valid(already_transformed_kwargs):
    ensure_expected_indexes_and_labels_present(
        already_transformed_kwargs,
        label_column_names=["cell_type", "tissue"],
        dataset_factory=CellXGeneDataset,
    )


def test_cellxgene_data_module_sequence_prediction(
    already_transformed_kwargs, gene2vec_fields, cellxgene_label_columns
):
    dm = CellXGeneDataModule(
        dataset_kwargs=already_transformed_kwargs,
        tokenizer=get_gene2vec_tokenizer(),
        label_columns=cellxgene_label_columns,
        fields=gene2vec_fields,
        transform_datasets=False,
        mlm=False,
        collation_strategy="sequence_classification",
    )
    dm.setup("fit")
    label = dm.train_dataset.label_columns[0]
    for batch in dm.train_dataloader():
        assert {*batch["labels"][label].numpy()}.issubset(dm.label_dict[label].values())
        break
    for batch in dm.val_dataloader():
        assert {*batch["labels"][label].numpy()}.issubset(dm.label_dict[label].values())
        break


def test_cellxgene_data_module_mlm(already_transformed_kwargs, gene2vec_fields):
    dm = CellXGeneDataModule(
        dataset_kwargs=already_transformed_kwargs,
        tokenizer=get_gene2vec_tokenizer(),
        fields=gene2vec_fields,
        transform_datasets=False,
        mlm=True,
    )
    dm.setup("fit")
    for batch in dm.train_dataloader():
        assert ((batch["labels"]["expressions"].numpy() == -100).mean(axis=0) > 0).all()
        break
    for batch in dm.val_dataloader():
        assert ((batch["labels"]["expressions"].numpy() == -100).mean(axis=0) > 0).all()
        break


def test_splits():
    check_splits(
        CellXGenePaths.root / "h5ad" / "cellxgene.h5ad",
        split_weights={"train": 0.4, "dev": 0.3, "test": 0.3},
        test_needle="placenta_179601",
        balancing_label="cell_type_ontology_term_id",
    )


def test_cellxgene_with_backed_data(already_transformed_kwargs, gene2vec_fields):
    dm0 = CellXGeneDataModule(
        dataset_kwargs=already_transformed_kwargs,
        tokenizer=get_gene2vec_tokenizer(),
        fields=gene2vec_fields,
        mlm=True,
        transform_datasets=False,
    )
    dm0.prepare_data()
    dm0.setup()
    assert isinstance(dm0.train_dataset.processed_data.X, csr_matrix)

    ds = CellXGeneDataset(
        processed_data_source=dm0.processed_data_file, split=None, backed="r"
    )

    assert isinstance(ds.processed_data.X, SparseDataset)
    for sample in ds:
        assert {*map(str, sample["expressions"])}.issubset(
            {*ds.get_vocab_for_field("expressions")}
        )
        assert {*sample["genes"]}.issubset({*ds.get_vocab_for_field("genes")})
        break


def test_cellxgene_downsample_with_backed_data(
    already_transformed_kwargs, gene2vec_fields
):
    dm = CellXGeneDataModule(
        dataset_kwargs={**already_transformed_kwargs, **{"backed": "r"}},
        tokenizer=get_gene2vec_tokenizer(),
        fields=gene2vec_fields,
        transform_datasets=False,
        limit_dataset_samples=10,
        mlm=True,
        num_workers=0,
        batch_size=1,
    )
    dm.prepare_data()
    dm.setup()
    all_batches = list(dm.train_dataloader())
    assert len(all_batches) == 10
