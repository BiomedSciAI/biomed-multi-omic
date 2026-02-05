#!/usr/bin/env python3
"""
Functions in this file are used to generate h5ad file for scp1884 dataset from downloaded tree from scp1884
(https://singlecell.broadinstitute.org/single_cell/study/SCP1884/human-cd-atlas-study-between-colon-and-terminal-ileum)
procedure calling these functions are written in the jupyter notebook
(https://github.ibm.com/A1KOSUGI/gsetests/blob/main/scp1884.ipynb).

"""

import gzip
import logging
import os
import re
import shutil

import anndata as ad
import pandas as pd
import scanpy as sc
from anndata import read_h5ad

from ..datasets_utils import obs_key_wise_subsampling

logging.basicConfig(
    level=logging.INFO,
    filename="scp1884_util.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_prefix(folder):
    """
    Find prefixes in the expression folders.

    Args:
    ----
        folder (path): path to the expression folder, which might be .../expression/12345/

    Returns:
    -------
        [str]: prefixes found in the folder
    """
    return [
        os.path.basename(f).replace("barcodes.tsv", "", -1)
        for f in folder.glob("*barcodes.tsv")
    ]


def read_metadata(dlfolder):
    """
    read metadata file (scp_metadata_combined.v2.txt) of scp1884.

    Args:
    ----
        dlfolder (Path/str): root folder whiere scp1884 files are downloaded
    """
    mdata = pd.read_csv(
        dlfolder / "metadata" / "scp_metadata_combined.v2.txt", skiprows=[1], sep="\t"
    )
    mdata = mdata.set_index("NAME")
    mdata["donor_id"] = mdata["donor_id"].apply(lambda x: str(x))
    return mdata


def fixup_features_tsv(folder, prefix):
    """
    fix *features.tsv to be read with sc.read_10x_mtx().

    Args:
    ----
        folder (Path/str): folder where the file resides
        prefix (str): prefix of the file
    """
    # original faeatures.tsv has only 2 columns while sc.read_10x_mtx expects 3rd column as feature_types..
    genes = pd.read_csv(folder / f"{prefix}features.tsv", header=None, sep="\t")
    if len(genes.columns) == 2:
        # shutil.copy2(folder / f"{prefix}features.tsv", folder / f"{prefix}features.tsv.original")
        # logger.info(f"{path} has only 2 column. assigning feature_types='Gene Expression'")
        genes[2] = "Gene Expression"
        genes.to_csv(
            folder / f"{prefix}features.tsv.gz",
            header=None,
            index=False,
            sep="\t",
            compression="gzip",
        )


def to_gzip(src, dst):
    """
    gzip a file.

    Args:
    ----
        src (Path/str): path to be gzipped
        dst (Path/str): destination path
    """
    if not os.path.exists(dst):
        with open(src, "rb") as f_in:
            with gzip.open(dst, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


def read_mtx(path, prefix):
    """
    read matrix file.

    Args:
    ----
        path (Path/str): path of the folder where the matrix file resides
        prefix (str): prefix of the set of files to be read
    """
    # fixup matrix(rename and gzip) features(feature_types and gzip) and barcodes(gzip)
    fixup_features_tsv(path, prefix)
    to_gzip(path / f"{prefix}raw.mtx", path / f"{prefix}matrix.mtx.gz")
    to_gzip(path / f"{prefix}barcodes.tsv", path / f"{prefix}barcodes.tsv.gz")

    return sc.read_10x_mtx(path=path, prefix=prefix)


def merge_metadata(adata, mdata):
    """
    merge metadata into the anndata read with read_mtx().

    Args:
    ----
        adata (Anndata): matrix anndata object
        metadata (Dataframe): metadata read with read_metadata()
    """
    adata.obs = (
        adata.obs.merge(
            mdata.reindex(index=adata.obs_names), left_index=True, right_index=True
        )
        .reset_index()
        .set_index("index")
    )
    return adata


def read_expression_files(dlfolder, metadata, cachefolder=None):
    """
    read files in expression folder of scp1884.

    Args:
    ----
        dlfolder (Path/str): path to the root folder of downloaded tree
        metadata (Dataframe): metadata read with read_metadata()
    """
    ret = {}
    if cachefolder is None:
        cachefolder = dlfolder / "h5ad"
    if not os.path.exists(cachefolder):
        os.mkdir(cachefolder)
    for o in filter(
        lambda x: len(x["prefix"]) > 0,
        (
            {"fld": x, "prefix": get_prefix(dlfolder / "expression" / x)}
            for x in filter(
                lambda x: os.path.isdir(dlfolder / "expression" / x),
                os.listdir(dlfolder / "expression"),
            )
        ),
    ):
        prefix = o["prefix"][0]  # ends with .
        path_h5ad = cachefolder / f"{prefix}h5ad"
        if os.path.exists(path_h5ad):
            logger.info(f"reading h5ad from {path_h5ad} for {prefix}...")
            ret[re.sub(r"\.$", "", prefix)] = read_h5ad(path_h5ad)
        else:
            logger.info(f"making h5ad in {path_h5ad} for {prefix}...")
            ad_mtx = read_mtx(path=dlfolder / "expression" / o["fld"], prefix=prefix)
            adata = merge_metadata(ad_mtx, metadata)
            adata.write_h5ad(path_h5ad, compression="gzip")
            ret[re.sub(r"\.$", "", prefix)] = adata
    return ret


def get_concat_h5ad(adatas, cachepath):
    """
    concatenate anndatas.

    Args:
    ----
        adatas ([key: anndata]): map of anndatas to be concatenated
        cachepath (Path/str): path to the cached h5ad
    """
    if os.path.exists(cachepath):
        logger.info(f"reading h5ad from {cachepath}...")
        ret = read_h5ad(cachepath)
    else:
        logger.info(f"concatinating h5ads into {cachepath}...")
        ret = ad.concat(adatas, join="outer", fill_value=0.0, label="batch")
        ret.write_h5ad(cachepath)
    return ret


def downsampleForTest(ad, N=15, obs_name_for_stratification="Celltype"):
    """
    Downsample and convert metadata of scp1884 h5ad for test files.

    Args:
    ----
        ad (anndata): anndata to be downsampled
        file_write (str): path to the output
        N (int, optional): downsample threshold. Defaults to 15.
    """
    # for gene index, gene ids are originally used so chage it to feature name
    # ad.var["feature_name_str"] = ad.var["feature_name"].astype(str)
    # ad.var = ad.var.set_index("feature_name_str")

    ad_downsample = obs_key_wise_subsampling(ad, obs_name_for_stratification, N=N)

    # metadata index has to have name "index"
    ad_downsample.obs.index.name = "index"

    # remove genes which no longer appear in any cells
    sc.pp.filter_genes(ad_downsample, min_cells=1)
    return ad_downsample
