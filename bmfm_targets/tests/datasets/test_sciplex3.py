import pytest

from bmfm_targets.datasets.sciplex3 import SciPlex3DataModule, SciPlex3Dataset
from bmfm_targets.tokenization import get_all_genes_tokenizer, load_tokenizer
from bmfm_targets.tokenization.resources import get_protein_coding_genes

from ..helpers import SciPlex3Paths, check_unk_levels_for_dm


@pytest.fixture(scope="module")
def dataset_kwargs():
    ds_kwargs = {
        "data_dir": SciPlex3Paths.root,
        "split_column": "split_random",
        "label_columns": ["cell_type"],
    }
    return ds_kwargs


def test_all_labels_accessible():
    all_labels = [
        "cell_type",
        "dose",
        "dose_character",
        "pathway",
        "pathway_level_1",
        "pathway_level_2",
        "product_dose",
        "product_name",
        "replicate",
        "target",
        "batch",
        "drug_dose_name",
        "cov_drug_dose_name",
        "cov_drug",
        "control",
        "SMILES",
    ]
    for label in all_labels:
        ds = SciPlex3Dataset(
            data_dir=SciPlex3Paths.root,
            split="train",
            split_column="split_random",
            label_columns=[label],
        )
        assert len(ds.label_dict) > 0

    ds_all_labels = SciPlex3Dataset(
        data_dir=SciPlex3Paths.root,
        split="train",
        split_column="split_random",
        label_columns=all_labels,
    )
    assert all(i in ds_all_labels[0].metadata for i in all_labels)


def test_load_sciplex3_from_test_resources_test_split(dataset_kwargs):
    ds = SciPlex3Dataset(**dataset_kwargs, split="test")

    assert ds[2].metadata == {
        "cell_name": "G05_F10_RT_BC_143_Lig_BC_18-1-0-3",
        "cell_type": "MCF7",
    }


def test_float_values_preserved_in_expressions(dataset_kwargs):
    ds = SciPlex3Dataset(**dataset_kwargs, transforms=[], split="test")

    mfi = ds[0]
    asints = [int(float(s)) for s in mfi["expressions"]]
    asfloats = [float(s) for s in mfi["expressions"]]
    assert not all(i == j for i, j in zip(asints, asfloats))


def test_load_sciplex3_from_test_resources_train_split(dataset_kwargs):
    ds = SciPlex3Dataset(**dataset_kwargs, split="train")

    assert ds[2].metadata == {
        "cell_name": "G01_F10_RT_BC_249_Lig_BC_374-1-0-3",
        "cell_type": "MCF7",
    }


def test_load_sciplex3_from_test_resources_dev_split(dataset_kwargs):
    ds = SciPlex3Dataset(**dataset_kwargs, split="dev")

    assert ds[2].metadata == {
        "cell_name": "A03_E09_RT_BC_379_Lig_BC_308-1-0-1",
        "cell_type": "MCF7",
    }


def test_sciplex3_data_module_sequence_prediction(
    dataset_kwargs, gene2vec_fields, sciplex3_label_columns
):
    dm = SciPlex3DataModule(
        dataset_kwargs=dataset_kwargs,
        tokenizer=get_all_genes_tokenizer(),
        fields=gene2vec_fields,
        label_columns=sciplex3_label_columns,
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


def test_sciplex3_data_module_mlm(dataset_kwargs, gene2vec_fields):
    dm = SciPlex3DataModule(
        dataset_kwargs=dataset_kwargs,
        tokenizer=get_all_genes_tokenizer(),
        fields=gene2vec_fields,
        mlm=True,
    )
    dm.setup("fit")

    for batch in dm.train_dataloader():
        assert ((batch["labels"]["expressions"].numpy() == -100).mean(axis=0) > 0).all()
        break
    for batch in dm.val_dataloader():
        assert ((batch["labels"]["expressions"].numpy() == -100).mean(axis=0) > 0).all()
        break


def test_sciplex3_data_module_mlm_with_nonbinned_expressions(
    dataset_kwargs, gene2vec_fields_regression_no_tokenization, sciplex3_label_columns
):
    dm = SciPlex3DataModule(
        dataset_kwargs={**dataset_kwargs, "transforms": []},
        tokenizer=get_all_genes_tokenizer(),
        fields=gene2vec_fields_regression_no_tokenization,
        label_columns=sciplex3_label_columns,
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


def test_data_module_limit_genes(
    dataset_kwargs, gene2vec_fields_regression_no_tokenization, sciplex3_label_columns
):
    dm = SciPlex3DataModule(
        dataset_kwargs={**dataset_kwargs, "transforms": []},
        tokenizer=load_tokenizer("gene2vec"),
        fields=gene2vec_fields_regression_no_tokenization,
        label_columns=sciplex3_label_columns,
        collation_strategy="sequence_classification",
        limit_genes="tokenizer",
    )
    dm.prepare_data()
    dm.setup("fit")
    mfi = dm.get_dataset_instance()[0]
    gene_vocab = dm.tokenizer.get_field_vocab("genes")

    assert all(g in gene_vocab for g in mfi["genes"])

    dm = SciPlex3DataModule(
        dataset_kwargs={**dataset_kwargs, "transforms": []},
        tokenizer=load_tokenizer("gene2vec"),
        fields=gene2vec_fields_regression_no_tokenization,
        label_columns=sciplex3_label_columns,
        collation_strategy="sequence_classification",
        limit_genes="protein_coding",
    )
    dm.prepare_data()
    dm.setup("fit")
    mfi = dm.get_dataset_instance()[0]
    assert not all(g in gene_vocab for g in mfi["genes"])

    pcg = get_protein_coding_genes()
    assert all(g in pcg for g in mfi["genes"])


def test_sciplex3_unk_levels(dataset_kwargs, gene2vec_fields):
    dm = SciPlex3DataModule(
        dataset_kwargs=dataset_kwargs,
        tokenizer=get_all_genes_tokenizer(),
        fields=gene2vec_fields,
        mlm=False,
        collation_strategy="language_modeling",
        limit_genes=None,
    )
    dm.prepare_data()
    dm.setup()
    check_unk_levels_for_dm(dm)
