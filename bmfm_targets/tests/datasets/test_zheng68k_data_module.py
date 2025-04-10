import numpy as np

from bmfm_targets.datasets.zheng68k import Zheng68kDataModule
from bmfm_targets.tests import helpers
from bmfm_targets.tokenization import load_tokenizer
from bmfm_targets.tokenization.resources import get_protein_coding_genes


def test_subsample_data(
    mock_data_dataset_kwargs_after_transform_without_labels, gene2vec_fields
):
    dm = Zheng68kDataModule(
        dataset_kwargs=mock_data_dataset_kwargs_after_transform_without_labels,
        transform_datasets=False,
        tokenizer=load_tokenizer("gene2vec"),
        fields=gene2vec_fields,
        limit_dataset_samples=5,
    )
    dm.prepare_data()
    dm.setup("fit")
    assert len(dm.train_dataset) == 5
    assert len(dm.dev_dataset) == 5


def test_rda_align_data(pl_mock_data_mlm_no_binning):
    dm = pl_mock_data_mlm_no_binning
    for batch in dm.train_dataloader():
        genes = batch["input_ids"][0, 0, :].to(int)
        expressions = batch["input_ids"][0, 1, :]
        assert genes[1] == dm.tokenizer.get_field_vocab("genes")["[S]"]
        assert genes[2] == dm.tokenizer.get_field_vocab("genes")["[T]"]
        np.testing.assert_allclose(expressions[2], np.log1p(2000))
        break


def test_too_large_subsample_doesnt_fail(
    mock_data_dataset_kwargs_after_transform_without_labels, gene2vec_fields
):
    dm = Zheng68kDataModule(
        dataset_kwargs=mock_data_dataset_kwargs_after_transform_without_labels,
        tokenizer=load_tokenizer("gene2vec"),
        fields=gene2vec_fields,
        limit_dataset_samples=100000,
        transform_datasets=False,
    )
    dm.prepare_data()
    dm.setup()


def test_data_module_limit_genes(
    mock_data_dataset_kwargs_after_transform_without_labels, gene2vec_fields
):
    dm = Zheng68kDataModule(
        dataset_kwargs=mock_data_dataset_kwargs_after_transform_without_labels,
        tokenizer=load_tokenizer("gene2vec"),
        fields=gene2vec_fields,
        limit_genes="tokenizer",
        transform_datasets=False,
    )
    dm.prepare_data()
    dm.setup("fit")
    mfi = dm.get_dataset_instance()[0]
    gene_vocab = dm.tokenizer.get_field_vocab("genes")
    assert all(g in gene_vocab for g in mfi["genes"])

    dm = Zheng68kDataModule(
        dataset_kwargs=mock_data_dataset_kwargs_after_transform_without_labels,
        tokenizer=load_tokenizer("gene2vec"),
        fields=gene2vec_fields,
        limit_genes="protein_coding",
        transform_datasets=False,
    )
    dm.prepare_data()
    dm.setup("fit")
    mfi = dm.get_dataset_instance()[0]
    assert not all(g in gene_vocab for g in mfi["genes"])

    pcg = get_protein_coding_genes()
    assert all(g in pcg for g in mfi["genes"])


def test_setup(mock_data_dataset_kwargs_after_transform_without_labels, gene2vec_fields):
    dm = Zheng68kDataModule(
        dataset_kwargs=mock_data_dataset_kwargs_after_transform_without_labels,
        tokenizer=load_tokenizer("gene2vec"),
        fields=gene2vec_fields,
        transform_datasets=False,
    )
    dm.prepare_data()
    dm.setup()
    assert len(dm.train_dataset) > 5


def test_subsample_data_shuffle(
    mock_data_label_columns, mock_data_dataset_kwargs_after_transform, gene2vec_fields
):
    dm = Zheng68kDataModule(
        dataset_kwargs=mock_data_dataset_kwargs_after_transform,
        tokenizer=load_tokenizer("gene2vec"),
        fields=gene2vec_fields,
        label_columns=mock_data_label_columns,
        limit_dataset_samples=5,
        transform_datasets=False,
        shuffle=False,
    )
    dm.prepare_data()
    dm.setup("fit")

    dm_shuffle = Zheng68kDataModule(
        dataset_kwargs=mock_data_dataset_kwargs_after_transform,
        tokenizer=load_tokenizer("gene2vec"),
        fields=gene2vec_fields,
        label_columns=mock_data_label_columns,
        limit_dataset_samples=5,
        transform_datasets=False,
        shuffle=True,
    )
    dm_shuffle.prepare_data()
    dm_shuffle.setup("fit")

    samples = dm.train_dataset.processed_data.obs_names.tolist()
    samples_shuffle = dm_shuffle.train_dataset.processed_data.obs_names.tolist()

    assert not np.array_equal(samples, samples_shuffle)


def test_rda_downsample_from_data_module(
    all_genes_fields_with_rda_regression_masking,
):
    tokenizer = load_tokenizer("all_genes")
    dm = Zheng68kDataModule(
        data_dir=helpers.MockTestDataPaths.root,
        processed_name=helpers.MockTestDataPaths.no_binning_name,
        transform_kwargs={"transforms": []},
        transform_datasets=True,
        dataset_kwargs={"source_h5ad_file_name":"mock_test_data.h5ad"},
        tokenizer=tokenizer,
        fields=all_genes_fields_with_rda_regression_masking,
        limit_dataset_samples=5,
        change_ratio=0.5,
        collation_strategy="language_modeling",
        sequence_order="sorted",
        mlm=True,
        rda_transform="downsample",
        max_length=512,  # to avoid downsample threshold
        pad_to_multiple_of=2,
        shuffle=False,
    )
    dm.prepare_data()
    dm.setup("fit")
    for batch in dm.train_dataloader():
        genes = batch["input_ids"][0, 0, :].to(int)
        expressions = batch["input_ids"][0, 1, :]
        label_expressions = batch["labels"]["expressions"].squeeze()
        assert genes[1] == tokenizer.get_field_vocab("genes")["[S]"]
        assert genes[2] == tokenizer.get_field_vocab("genes")["[T]"]
        assert expressions[1] <= expressions[2]
        assert 0 < (label_expressions == -100).to(float).mean() < 1
        modified_mask_token_id = -(tokenizer.get_field_vocab("genes")["[MASK]"] + 1)
        assert 0 < (expressions == float(modified_mask_token_id)).to(float).mean() < 1
        break


def test_rda_downsample_with_sequence_classification_collation(
    all_genes_fields_with_rda_regression_masking, mock_data_label_columns
):
    tokenizer = load_tokenizer("all_genes")
    dm = Zheng68kDataModule(
        data_dir=helpers.MockTestDataPaths.root,
        processed_name=helpers.MockTestDataPaths.no_binning_name,
        transform_kwargs={"transforms": []},
        dataset_kwargs={"source_h5ad_file_name":"mock_test_data.h5ad"},
        transform_datasets=True,
        tokenizer=tokenizer,
        fields=all_genes_fields_with_rda_regression_masking,
        label_columns=mock_data_label_columns,
        limit_dataset_samples=5,
        collation_strategy="sequence_classification",
        sequence_order="sorted",
        rda_transform="downsample",
        mlm=False,
        max_length=512,  # to avoid downsample threshold
        pad_to_multiple_of=2,
        shuffle=False,
    )
    dm.prepare_data()
    dm.setup("fit")
    for batch in dm.train_dataloader():
        genes = batch["input_ids"][0, 0, :].to(int)
        expressions = batch["input_ids"][0, 1, :]
        labels = batch["labels"]["celltype"].squeeze()
        assert genes[1] == tokenizer.get_field_vocab("genes")["[S]"]
        assert genes[2] == tokenizer.get_field_vocab("genes")["[T]"]
        assert expressions[1] <= expressions[2]
        assert all(i < len(dm.label_dict["celltype"]) for i in labels)
        modified_mask_token_id = -(tokenizer.get_field_vocab("genes")["[MASK]"] + 1)
        assert (expressions == float(modified_mask_token_id)).to(float).mean() == 0
        break


def test_max_counts_calculated_correctly(all_genes_fields_with_rda_regression_masking):
    from bmfm_targets.training import sample_transforms

    tokenizer = load_tokenizer("all_genes")
    dm = Zheng68kDataModule(
        data_dir=helpers.MockTestDataPaths.root,
        dataset_kwargs={"source_h5ad_file_name":"mock_test_data.h5ad"},
        tokenizer=tokenizer,
        fields=all_genes_fields_with_rda_regression_masking,
    )
    dm.prepare_data()
    dm.setup("fit")
    dataset = dm.get_dataset_instance()
    max_len = 512
    unlimited_max_counts = dataset.max_counts()
    max_counts_up_to_length = dataset.max_counts(max_len)
    max_counts_measured = max(
        [sum(dataset[i]["expressions"]) for i in range(len(dataset))]
    )
    assert max_counts_measured == unlimited_max_counts
    max_counts_limited_measured = max(
        [
            sum(
                sample_transforms.sort_by_field(dataset[i], "expressions")[
                    "expressions"
                ][:max_len]
            )
            for i in range(len(dataset))
        ]
    )
    assert max_counts_limited_measured == max_counts_up_to_length


def test_rda_auto_align_from_data_module(
    all_genes_fields_with_regression_masking,
):
    tokenizer = load_tokenizer("all_genes")
    dm = Zheng68kDataModule(
        data_dir=helpers.MockTestDataPaths.root,
        processed_name=helpers.MockTestDataPaths.no_binning_name,
        transform_kwargs={"transforms": []},
        dataset_kwargs={"source_h5ad_file_name":"mock_test_data.h5ad"},
        transform_datasets=True,
        tokenizer=tokenizer,
        fields=all_genes_fields_with_regression_masking,
        limit_dataset_samples=8,
        collation_strategy="language_modeling",
        mlm=True,
        rda_transform="auto_align",
        max_length=20,
        pad_to_multiple_of=2,
    )
    dm.prepare_data()
    dm.setup("fit")
    dataset_max_reads = int(dm.train_dataset.max_counts())
    t_expressions = [batch["input_ids"][:, 1, 2] for batch in dm.train_dataloader()]
    np.testing.assert_allclose(t_expressions, np.log1p(dataset_max_reads))


def test_rda_equal_from_data_module(
    all_genes_fields_with_rda_regression_masking,
):
    tokenizer = load_tokenizer("all_genes")
    dm = Zheng68kDataModule(
        data_dir=helpers.MockTestDataPaths.root,
        processed_name=helpers.MockTestDataPaths.no_binning_name,
        transform_kwargs={"transforms": []},
        dataset_kwargs={"source_h5ad_file_name":"mock_test_data.h5ad"},
        transform_datasets=True,
        tokenizer=tokenizer,
        fields=all_genes_fields_with_rda_regression_masking,
        limit_dataset_samples=5,
        change_ratio=0.5,
        collation_strategy="language_modeling",
        sequence_order="sorted",
        mlm=True,
        rda_transform="equal",
        max_length=512,  # to avoid downsample threshold
        pad_to_multiple_of=2,
        shuffle=False,
    )
    dm.prepare_data()
    dm.setup("fit")
    for batch in dm.train_dataloader():
        expressions = batch["input_ids"][0, 1, :]
        assert expressions[1] == expressions[2]
        break
