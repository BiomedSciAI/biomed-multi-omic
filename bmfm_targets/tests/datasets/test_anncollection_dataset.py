import pytest

from bmfm_targets.datasets.anncollection import (
    AnnCollectionDataModule,
    AnnCollectionDataset,
    get_ann_collection,
)
from bmfm_targets.datasets.data_conversion.litdata_indexing import build_index
from bmfm_targets.tokenization import get_gene2vec_tokenizer

from ..helpers import (
    AnnCollectionDatasetPaths,
    ChangEpiPaths,
    scIBD300kPaths,
)


def create_index(dataset_dir, index_dir, join_obs="inner"):
    collection = get_ann_collection(input_dir=str(dataset_dir), join_obs=join_obs)
    n_cells = collection.n_obs
    build_index(output_dir=str(index_dir), index=range(0, n_cells), chunk_size=500)


@pytest.fixture(scope="module")
def _anncollection_index():
    AnnCollectionDatasetPaths.root.mkdir(exist_ok=True)
    index_dir = AnnCollectionDatasetPaths.root / "Index"
    index_dir.mkdir(exist_ok=True)
    dataset_dir = AnnCollectionDatasetPaths.dataset
    create_index(dataset_dir=dataset_dir, index_dir=index_dir / "train")
    create_index(dataset_dir=dataset_dir, index_dir=index_dir / "dev")


@pytest.fixture(scope="module")
def _anncollection_index_w_label():
    AnnCollectionDatasetPaths.root.mkdir(exist_ok=True)
    index_dir = AnnCollectionDatasetPaths.root / "Index_labelled"
    index_dir.mkdir(exist_ok=True)
    dataset_dir = AnnCollectionDatasetPaths.root / "Data_labelled"
    dataset_dir.mkdir(exist_ok=True)
    for cl, nm in [(scIBD300kPaths, "scIBD300k"), (ChangEpiPaths, "changepi")]:
        if not (dataset_dir / f"{nm}.h5ad").exists():
            (dataset_dir / f"{nm}.h5ad").symlink_to(cl.root / "h5ad" / f"{nm}.h5ad")
    create_index(
        dataset_dir=dataset_dir, index_dir=index_dir / "train", join_obs="outer"
    )
    create_index(dataset_dir=dataset_dir, index_dir=index_dir / "dev", join_obs="outer")


def make_dataset_kwargs(expose_zeros):
    return {
        "dataset_dir": str(AnnCollectionDatasetPaths.dataset),
        "index_dir": str(AnnCollectionDatasetPaths.root / "Index"),
        "expose_zeros": expose_zeros,
    }


def make_dataset_kwargs_w_label(expose_zeros):
    return {
        "dataset_dir": str(AnnCollectionDatasetPaths.root / "Data_labelled"),
        "index_dir": str(AnnCollectionDatasetPaths.root / "Index_labelled"),
        "expose_zeros": expose_zeros,
        "label_columns": ["celltype", "Final_CellType"],
    }


@pytest.fixture()
def dataset_input(gene2vec_fields):
    tokenizer = get_gene2vec_tokenizer()
    pars = {
        "tokenizer": tokenizer,
        "batch_size": 2,
        "fields": gene2vec_fields,
        "num_workers": 0,
        "mlm": True,
        "collation_strategy": "language_modeling",
    }
    return pars


@pytest.mark.usefixtures("_anncollection_index")
@pytest.mark.parametrize("expose_zeros", [(None), ("all")])
def test_anndata_dataset(expose_zeros, dataset_input):
    dataset_input["dataset_kwargs"] = make_dataset_kwargs(expose_zeros)
    datamodule = AnnCollectionDataModule(**dataset_input)
    datamodule.prepare_data()
    datamodule.setup("fit")
    for batch in datamodule.train_dataloader():
        assert batch is not None
        break


@pytest.mark.usefixtures("_anncollection_index")
def test_expose_zeros():
    all_zeros_ds = AnnCollectionDataset(**make_dataset_kwargs("all"), split="train")
    no_zeros_ds = AnnCollectionDataset(**make_dataset_kwargs(None), split="train")

    all_zeros_item = all_zeros_ds[1]
    no_zeros_item = no_zeros_ds[1]

    assert len(all_zeros_item.data["genes"]) > len(no_zeros_item.data["genes"])
    assert len(all_zeros_item.data["expressions"]) > len(
        no_zeros_item.data["expressions"]
    )


@pytest.fixture()
def dataset_input_w_label(gene2vec_fields):
    tokenizer = get_gene2vec_tokenizer()
    pars = {
        "tokenizer": tokenizer,
        "batch_size": 2,
        "fields": gene2vec_fields,
        "num_workers": 0,
        "mlm": False,
        "collation_strategy": "sequence_classification",
    }
    return pars


@pytest.mark.usefixtures("_anncollection_index_w_label")
@pytest.mark.parametrize("expose_zeros", [(None), ("all")])
def test_anndata_dataset_w_label(expose_zeros, dataset_input_w_label):
    dataset_input_w_label["dataset_kwargs"] = make_dataset_kwargs_w_label(expose_zeros)
    datamodule = AnnCollectionDataModule(**dataset_input_w_label)
    datamodule.prepare_data()
    datamodule.setup("fit")

    assert "celltype" in datamodule.label_dict
    assert "Final_CellType" in datamodule.label_dict

    for batch in datamodule.train_dataloader():
        assert batch is not None
        break
