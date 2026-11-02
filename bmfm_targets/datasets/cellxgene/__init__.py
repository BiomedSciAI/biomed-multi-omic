"""
The single_cell_rna package consists of modules for processing single-cell
RNA-seq data and converting them to pytorch datasets for use in training neural
networks.
"""
from .cellxgene_dataset import CellXGeneDataset, CellXGeneDataModule
from .cellxgene_nexus_dataset import CellXGeneNexusDataset, CellXGeneNexusDataModule

# The SOMA names need cellxgene_census[experimental]; resolve them lazily so other names import without it.
_LAZY_SUBMODULES = {
    "CellXGeneSOMADataset": ".cellxgene_soma_dataset",
    "CellXGeneSOMADataModule": ".cellxgene_soma_dataset",
    "create_litdata_index_for_dataset_split": ".cellxgene_soma_utils",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        from importlib import import_module

        return getattr(import_module(_LAZY_SUBMODULES[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CellXGeneDataset",
    "CellXGeneDataModule",
    "CellXGeneSOMADataset",
    "CellXGeneSOMADataModule",
    "CellXGeneNexusDataset",
    "CellXGeneNexusDataModule",
    "create_litdata_index_for_dataset_split",
]
