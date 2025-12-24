import abc
import datetime
import inspect
import json
import logging
import pickle
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from rnanorm import TPM
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def lookup_name(transform_name):
    """
    Lookup the name of the transform and return the class object.

    Args:
    ----
        transform_name: The name of the transform to lookup.

    Returns:
    -------
        class_object: The class object of the transform.

    Raises:
    ------
        ValueError: If the transform name is not found.
    """
    current_module = inspect.getmodule(lookup_name)
    class_object = getattr(current_module, transform_name)
    if issubclass(class_object, BaseSCTransform):
        return class_object
    raise ValueError("Encountered unknown transform name: f{transform_name}")


class BaseSCTransform(abc.ABC):
    """Base class for all SCTransform classes."""

    @abc.abstractmethod
    def __call__(self, adata: AnnData) -> Mapping[str, Any]:
        ...


def var_index_is_ensembl(adata: AnnData) -> bool:
    """
    Check if the given variable index is a symbol.

    Args:
    ----
        adata: The input AnnData object.
        var: The variable index to check.

    Returns:
    -------
        bool: True if the variable index is a symbol, False otherwise.

    Raises:
    ------
        ValueError: If the input data is not an AnnData object.
    """
    if not isinstance(adata, AnnData):
        raise ValueError("Input data must be an AnnData object.")

    # Define a regular expression for both human and mouse Ensembl gene IDs
    ensembl_pattern = re.compile(r"^ENSG|^ENSMUSG")

    # Check if most of the first 10 entries match the Ensembl pattern
    if adata.n_vars < 10:
        raise ValueError("AnnData has less than 10 variables/genes.")

    is_ensembl = adata.var_names[:10].str.match(ensembl_pattern).sum() > 8

    return is_ensembl


class DupGeneStrategy(Enum):
    RENAME = "rename"
    AGGREGATE = "aggregate"
    REMOVE = "remove"


class RenameGenesTransform(BaseSCTransform):
    """

    A class for renaming genes in an AnnData object.


    Attributes
    ----------
        gene_map: A dictionary mapping old gene names to new gene names.
    """

    def __init__(self, gene_map: Mapping[str, str] | None = None):
        """
        Initialize the RenameGenesTransform class.


        Args:
        ----
            gene_map: A dictionary mapping old gene names to new gene names.
        """
        self.gene_map = gene_map

    def __call__(self, adata: AnnData) -> Mapping[str, AnnData]:
        """
        Apply the rename genes transformation to the provided AnnData object.


        Args:
        ----
            adata: The input AnnData object.


        Returns:
        -------
            AnnData: The transformed AnnData object.


        Raises:
        ------
            ValueError: If the input data is not an AnnData object.
            ValueError: If the transformation fails.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("Input data must be an AnnData object.")

        if self.gene_map is None:
            mapping_dict = {gene: gene.split("_ENSG")[0] for gene in adata.var.index}
        else:
            mapping_dict = {
                gene: self.gene_map[gene]
                for gene in adata.var.index
                if gene in self.gene_map
            }
        # Apply the rename genes transformation
        try:
            adata.var.index = adata.var.index.map(mapping_dict)
            # messing with this index triggers a known limitation with anndata objects
            # see https://github.com/scverse/anndata/issues/452
            if adata.var.index.name in adata.var.columns:
                adata.var.pop(adata.var.index.name)
            unique_gene_ids = []
            indices_of_duplicates = []

            for i, gene_id in enumerate(adata.var_names):
                if gene_id not in unique_gene_ids:
                    unique_gene_ids.append(gene_id)
                else:
                    indices_of_duplicates.append(i)

            # TODO: here we are removing the duplicate genes. Not sure if this is the best way to do it. If we dont do this KeepGenesTransform will fail
            mask = np.ones(len(adata.var_names), dtype=bool)
            mask[indices_of_duplicates] = False
            adata = adata[:, mask]

        except Exception as e:
            raise ValueError(
                "Error occurred during rename genes transformation:", str(e)
            )

        # Return the transformed AnnData object
        return {"adata": adata}


class StandardizeGeneNamesTransform(BaseSCTransform):
    """

    A class for renaming genes in an AnnData object.


    Attributes
    ----------
        gene_map: A dictionary mapping old gene names to new gene names.
    """

    def __init__(
        self,
        gene_map: Mapping[str, str] | None = None,
        dup_gene_strategy: DupGeneStrategy = DupGeneStrategy.REMOVE,
    ):
        """
        Initialize the RenameGenesTransform class.


        Args:
        ----
            gene_map: A dictionary mapping old gene names to new gene names.
            dup_gene_strategy:
                "remove": delete duplicated gene/var columns (keep first)
                "rename": rename deplicated gene/var columns (first unchanged)
                "aggregate": summing expression values and then remove
        """
        self.is_default_gene_map = False
        if not gene_map:
            # Load the dictionary from a JSON file
            from pathlib import Path

            file_path = (
                Path(__file__).resolve().parent
                / "protein_coding_gene_mapping_uppercase_hgnc_2024_08_23.json"
            )
            gene_map = json.load(file_path.open())
            self.is_default_gene_map = True

        self.gene_map = gene_map
        self.dup_gene_strategy = dup_gene_strategy

    def match_above_threshold(self, var_names, threshold=0.3):
        match_count = sum([1 for s in var_names if s in self.gene_map])
        match_percentage = match_count / len(self.gene_map)
        return match_percentage >= threshold

    def __call__(self, adata: AnnData) -> Mapping[str, AnnData]:
        """
        Apply the rename genes transformation to the provided AnnData object.


        Args:
        ----
            adata: The input AnnData object.


        Returns:
        -------
            AnnData: The transformed AnnData object.


        Raises:
        ------
            ValueError: If the input data is not an AnnData object.
            ValueError: If the transformation fails.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("Input data must be an AnnData object.")

        if var_index_is_ensembl(adata) and self.is_default_gene_map:
            raise ValueError(
                "Input AnnData must use gene symbols as variable index when using the default gene map."
            )

        if not self.match_above_threshold(adata.var_names):
            unique_mask = ~adata.var.index.duplicated(keep="first")
            adata = adata[:, unique_mask]
            return {"adata": adata}

        num_genes = adata.n_vars
        if self.is_default_gene_map:
            adata.var_names = adata.var_names.str.upper()

        adata.var["new_index"] = adata.var.index.map(self.gene_map)
        num_na = adata.var["new_index"].isna().sum()
        unmapped_gene_names = adata.var.index[adata.var["new_index"].isna()]
        adata = adata[:, adata.var["new_index"].notna()]

        if adata.var["new_index"].duplicated().any():
            if self.dup_gene_strategy == DupGeneStrategy.REMOVE:
                adata.var = adata.var.loc[
                    ~adata.var["new_index"].duplicated(keep="first")
                ]
                adata = adata[:, adata.var.index.isin(adata.var["new_index"])]
            elif self.dup_gene_strategy == DupGeneStrategy.RENAME:
                adata.var["new_index"] = (
                    pd.Series(adata.var["new_index"])
                    .astype(str)
                    .where(
                        ~adata.var["new_index"].duplicated(),
                        lambda x: x
                        + "_"
                        + (x.index - x.index.duplicated(keep="first")).astype(str),
                    )
                )
            elif self.dup_gene_strategy == DupGeneStrategy.AGGREGATE:
                raise NotImplementedError("This function is not implemented yet.")
            else:
                raise ValueError(f"Unknown dup_gene_strategy: {self.dup_gene_strategy}")

        if adata.var["new_index"].duplicated().any():
            raise ValueError("Failed to resolve duplicate values in 'new_index'.")
        adata.var.index = adata.var["new_index"]
        adata.var = adata.var.drop(columns=["new_index"])

        text_gene_num = (
            f"Number of genes: Original = {num_genes}, Not found in map = {num_na}"
        )
        logger.info(
            f"### In [{self.__class__.__name__}] ###\n{text_gene_num}\n"
            f"Unmapped genes: {unmapped_gene_names.tolist()}"
        )

        adata.uns["number_of_genes"] = text_gene_num
        adata.uns["unmapped_genes"] = unmapped_gene_names.tolist()

        return {"adata": adata}


class KeepGenesTransform(BaseSCTransform):
    """
    A class for keeping genes in an AnnData object.


    Attributes
    ----------
        genes_to_keep: A list of genes to keep or names of the file with a list of genes
    """

    def __init__(self, genes_to_keep: list[str] | str | None = None):
        """
        Initialize the KeepGenesTransform class.

        Args:
        ----
            genes_to_keep: A list of genes to keep or names of the file with a list of genes
        """
        if isinstance(genes_to_keep, str):
            with open(genes_to_keep) as file:
                self.genes_to_keep = file.readlines()
                self.genes_to_keep = [line.strip() for line in self.genes_to_keep]
        else:
            self.genes_to_keep = genes_to_keep

    def __call__(self, adata: AnnData) -> Mapping[str, AnnData]:
        """
        Apply the keep genes transformation to the provided AnnData object.


        Args:
        ----
            adata: The input AnnData object.


        Returns:
        -------
            AnnData: The transformed AnnData object.


        Raises:
        ------
            ValueError: If the input data is not an AnnData object.
            ValueError: If the transformation fails.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("Input data must be an AnnData object.")

        if self.genes_to_keep is None:
            keep_genes = adata.var.index
        else:
            keep_genes = self.genes_to_keep
        # Apply the keep genes transformation
        try:
            # check if genes are in adata.var.index
            logger.info("in KeepGenes: ")
            logger.info(f"adata shape before = {adata.to_df().shape}")
            data_genes = list(adata.var.index)
            filtered_genes = [gene for gene in keep_genes if gene in data_genes]
            adata = adata[:, filtered_genes]
            logger.info(f"adata shape after = {adata.to_df().shape}")

        except Exception as e:
            raise ValueError("Error occurred during keep genes transformation:", str(e))

        # Return the transformed AnnData object
        return {"adata": adata}


class KeepVocabGenesTransform(KeepGenesTransform):
    """A class for keeping genes in an AnnData object by only using genes in the tokenizer vocabulary."""

    def __init__(self, vocab_identifier: str | None = None):
        """
        Initialize the KeepVocabGenesTransform class.

        Args:
        ----
            vocab_identifier: name of packaged tokenizer ("gene2vec" or "all_genes")
        """
        from bmfm_targets.tokenization import load_tokenizer  # to avoid circular import

        self.genes_to_keep = load_tokenizer(vocab_identifier).get_field_vocab("genes")


class KeepExplanatoryAndTargetGenesTransform(KeepGenesTransform):
    """A class for keeping genes in an AnnData object by only using genes in explanatory gene sets and regression labels. This class is working with vocab_identifier so it no use of calling this class independently."""

    def __init__(
        self,
        explanatory_gene_set: list[str] | str = None,
        regression_label_columns: list[str] = None,
    ):
        """
        Initialize the KeepExplanatoryAndTargetGenesTransform class.

        Args:
        ----
            explanatory_gene_set: A list of genes to keep or names of the file with a list of genes which are used for explanatory values of regression
            regression_label_columns: A list of genes to extract expression levels for regression targets
        """
        explanatory_gene_set_ = None

        if explanatory_gene_set is not None:
            if isinstance(explanatory_gene_set, str):
                logger.info("Loading TFs from " + str(explanatory_gene_set))
                tf_list = pd.read_csv(explanatory_gene_set, header=None)
                explanatory_gene_set_ = list(tf_list[0])
            else:
                explanatory_gene_set_ = explanatory_gene_set.copy()
        else:
            raise ValueError(
                "Error occurred during keep genes transformation: explanatory_gene_set must not be None"
            )

        if regression_label_columns is not None:
            assert not set(regression_label_columns).issubset(explanatory_gene_set_)
            # remove intersection between regression_label_columns and explanatory_gene_set
            unique_regression_label_columns = list(
                set(regression_label_columns) - set(explanatory_gene_set_)
            )
            explanatory_gene_set_ += unique_regression_label_columns
        else:
            raise ValueError(
                "Error occurred during keep genes transformation: regression_label_columns must not be None"
            )

        self.genes_to_keep = explanatory_gene_set_


class MoveExpressionsToLabels(BaseSCTransform):
    """A class for keeping genes in an AnnData object by only using genes in explanatory gene sets and regression labels. This class is working with KeepExplanatoryAndTargetGenesTransform, and extract expression levels for regression target genes. Finally target genes themselves are removed to exclude target genes from explanatory genes."""

    def __init__(
        self,
        regression_label_columns: str = None,
    ):
        """
        Initialize the MoveExpressionsToLabels class.

        Args:
        ----
            regression_label_columns: A list of genes to extract expression levels for regression targets
        """
        self.regression_label_columns = regression_label_columns

    def __call__(self, adata: AnnData) -> Mapping[str, AnnData]:
        """
        Apply the keep genes transformation to the provided AnnData object.


        Args:
        ----
            adata: The input AnnData object.


        Returns:
        -------
            AnnData: The transformed AnnData object.


        Raises:
        ------
            ValueError: If the input data is not an AnnData object.
            ValueError: If the transformation fails.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("Input data must be an AnnData object.")

        if self.regression_label_columns is None:
            raise ValueError(
                "Error occurred during keep genes transformation: regression_label_columns must not be None"
            )

        try:
            regression_label_expressions_df = adata[
                :, self.regression_label_columns
            ].to_df()
            adata.obs = pd.concat([adata.obs, regression_label_expressions_df], axis=1)
            # remove genes in regression_label_columns
            filtered_genes = [
                gene
                for gene in adata.var.index
                if gene not in self.regression_label_columns
            ]
            adata = adata[:, filtered_genes]

        except Exception as e:
            raise ValueError("Error occurred during keep genes transformation:", str(e))

        # Return the transformed AnnData object
        return {"adata": adata}


class ChangeExpressionLevelTransform(BaseSCTransform):
    """A class for change expression level of specific genes to conduct in-sillico perturbation in sequence classification."""

    def __init__(
        self,
        gene_perturbations: dict = None,
    ):
        """
        Initialize the ChangeExpressionLevelTransform class.

        Args:
        ----
            regression_label_columns: A list of genes to extract expression levels for regression targets
        """
        self.gene_perturbations = gene_perturbations

    def __call__(self, adata: AnnData) -> Mapping[str, AnnData]:
        """
        Apply the keep genes transformation to the provided AnnData object.


        Args:
        ----
            adata: The input AnnData object.


        Returns:
        -------
            AnnData: The transformed AnnData object.


        Raises:
        ------
            ValueError: If the input data is not an AnnData object.
            ValueError: If the transformation fails.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("Input data must be an AnnData object.")

        if self.gene_perturbations is None:
            raise ValueError(
                "Error occurred during ChangeExpressionLevel transformation: gene_perturbations must not be None"
            )

        try:
            for gene, direction in self.gene_perturbations.items():
                if direction == "positive":
                    expressions = adata[:, gene]
                    perturb_value = np.max(expressions.X[:, 0])
                elif direction == "negative":
                    expressions = adata[:, gene]
                    perturb_value = np.min(expressions.X[:, 0])
                else:
                    raise ValueError(
                        "Error occurred during ChangeExpressionLevel transformation: illegal perturbation (must be positive or negative)",
                        direction,
                    )

                logger.info(
                    "Change expresion value of "
                    + gene
                    + " to "
                    + str(perturb_value)
                    + " for in-sillico perturbation"
                )
                adata[:, gene] = perturb_value

        except Exception as e:
            raise ValueError(
                "Error occurred during ChangeExpressionLevel transformation:", str(e)
            )

        # Return the transformed AnnData object
        return {"adata": adata}


class TPMNormalizationTransform(BaseSCTransform):
    """
    A class for TPM normalization on data. Used for bulkRNA normalization.

    Args:
    ----
        gtf_file: path to the GTF (Gene Transfer Format) file of the corresponding genom
        filter_genes_min_value: minimal value of the expression after transformation to be counted in for filtering (used together with filter_genes_min_percentage)
        filter_genes_min_percentage: gene will be dropped if percent of samples with value above 'filter_genes_min_value' is less then this argument

    Returns:
    -------
        AnnData: The transformed AnnData object.
    """

    def __init__(
        self,
        gtf_file: str | None = None,
        filter_genes_min_value: float | None = None,
        filter_genes_min_percentage: float | None = None,
    ):
        """
        Initialize the PMNormalizationTransform class.


        Args:
        ----
            gtf_file: path to the GTF (Gene Transfer Format) file of the corresponding genom
            filter_genes_min_value: minimal value of the expression after transformation to be counted in for filtering (used together with filter_genes_min_percentage)
            filter_genes_min_percentage: gene will be dropped if percent of samples with value above 'filter_genes_min_value' is less then this argument

        Raises:
        ------
            ValueError: If the GTF file is misssin.
            ValueError: If the initialization of TPM fails.
        """
        if gtf_file is None:
            raise ValueError(
                "GTF file is not provided, TPM normalization can't be applied"
            )

        try:
            self.tpm_transformer = TPM(gtf=gtf_file).set_output(transform="pandas")
        except Exception as e:
            raise ValueError(
                "Error occurred during TPM normalization initialization:", str(e)
            )

        self.filter_genes_min_value = filter_genes_min_value
        self.filter_genes_min_percentage = filter_genes_min_percentage

    def __call__(self, adata: AnnData) -> Mapping[str, Any]:
        """
        Apply the TPM normalization transformation to the provided AnnData object.

        Args:
        ----
            adata: The input AnnData object.

        Returns:
        -------
            AnnData: The transformed AnnData object.

        Raises:
        ------
            ValueError: If the input data is not an AnnData object.
            ValueError: If the transformation fails.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("Input data must be an AnnData object.")

        try:
            # remove all 0 rows from the dataframe
            nonzero_mask = np.any(adata.to_df() != 0, axis=1)
            adata_filtered = adata[nonzero_mask].copy()
            df_tpm = self.tpm_transformer.fit_transform(adata_filtered.to_df())
            # filter genes that are not in gtf file
            df_tpm = df_tpm.dropna(axis=1)

            # keep genes that have more that self.filter_genes_min_percentage of samples with values > self.filter_genes_min_value
            if (not self.filter_genes_min_value is None) and (
                self.filter_genes_min_value > 0
            ):
                ind_cols = (df_tpm > self.filter_genes_min_value).sum(
                    axis=0
                ) >= self.filter_genes_min_percentage * df_tpm.shape[0]
                df_tpm = df_tpm.loc[:, ind_cols]

            matrix = csr_matrix(df_tpm.values)
            adata = sc.AnnData(X=matrix, obs=adata_filtered.obs)
            adata.obs_names = df_tpm.index
            adata.var_names = df_tpm.columns

        except Exception as e:
            raise ValueError(
                "Error occurred during TPM normalization transformation:", str(e)
            )

        return {"adata": adata}


class FilterCellsTransform(BaseSCTransform):
    """
    A class for applying the scanpy.pp.filter_cells function as a
    transformation.

    Attributes
    ----------
        min_counts: Minimum count threshold for filtering cells.
        min_genes: Minimum gene threshold for filtering cells.
        max_counts: Maximum count threshold for filtering cells.
        max_genes: Maximum gene threshold for filtering cells.
    """

    def __init__(
        self,
        min_counts: int | None = None,
        min_genes: int | None = None,
        max_counts: int | None = None,
        max_genes: int | None = None,
    ):
        """
        Initialize the FilterCellsTransform class.

        Args:
        ----
            min_counts: Minimum count threshold for filtering cells.
            min_genes: Minimum gene threshold for filtering cells.
            max_counts: Maximum count threshold for filtering cells.
            max_genes: Maximum gene threshold for filtering cells.
        """
        self.min_counts = min_counts
        self.min_genes = min_genes
        self.max_counts = max_counts
        self.max_genes = max_genes

    def __call__(self, adata: AnnData) -> Mapping[str, Any]:
        """
        Apply the filter cells transformation to the provided AnnData
        object.

        Args:
        ----
            adata: The input AnnData object.

        Returns:
        -------
            AnnData: The transformed AnnData object.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("Input data must be an AnnData object.")
        try:
            logger.info("in FilterCellsTransform: ")
            logger.info(f"adata shape before = {adata.shape}")
            sc.pp.filter_cells(
                adata,
                min_counts=self.min_counts,
                min_genes=self.min_genes,
                max_counts=self.max_counts,
                max_genes=self.max_genes,
            )
            logger.info(f"adata shape after = {adata.shape}")

        except Exception as e:
            raise e
        return {"adata": adata}


class QcMetricsTransform(BaseSCTransform):
    """
    A class for applying the sc.pp.ccalculate_qc_metrics as a
    transformation.

    Attributes
    ----------
        pct_counts_mt: Threshold count of pct_counts_mt for filtering genes.
    """

    def __init__(
        self,
        pct_counts_mt: int | None = None,
        total_counts_iqr_scale: float | None = None,
    ):
        """
        Initialize the FilterGenesTransform class.

        Args:
        ----
            min_counts: Minimum count threshold for filtering genes.
            total_counts_iqr_scale: IQR scale for removing very high and low count samples
               good default value is 1.5
        """
        self.pct_counts_mt = pct_counts_mt
        self.total_counts_iqr_scale = total_counts_iqr_scale

    def __call__(self, adata: AnnData) -> Mapping[str, AnnData]:
        """
        Apply the qc_metrics transformation to the provided input data.

        Args:
        ----
            data:  The input AnnData object.

        Returns:
        -------
            AnnData: The transformed AnnData object.

        Raises:
        ------
            ValueError: If the input data is not an AnnData object.
            ValueError: If the transformation fails.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("Input data must be an AnnData object.")

        # Apply the filter genes transformation
        try:
            adata.var["mt"] = adata.var_names.str.startswith("MT-")
            sc.pp.calculate_qc_metrics(
                adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
            )
            if self.pct_counts_mt is not None:
                adata = adata[adata.obs.pct_counts_mt < self.pct_counts_mt, :]
            if self.total_counts_iqr_scale is not None:
                iqr_scale = self.total_counts_iqr_scale
                counts = adata.obs["total_counts"]

                q1 = np.percentile(counts, 25)
                q3 = np.percentile(counts, 75)
                iqr = q3 - q1

                lower = q1 - iqr_scale * iqr
                upper = q3 + iqr_scale * iqr

                mask = (counts >= lower) & (counts <= upper)
                adata = adata[mask].copy()
        except Exception as e:
            raise ValueError(
                "Error occurred during applying qc_metrics transformation:", str(e)
            )

        # Return the transformed AnnData object
        return {"adata": adata}


class FilterGenesTransform(BaseSCTransform):
    """
    A class for applying the scanpy.pp.filter_genes function as a
    transformation.

    Attributes
    ----------
        min_counts: Minimum count threshold for filtering genes.
        min_cells: Minimum cell threshold for filtering genes.
        max_counts: Maximum count threshold for filtering genes.
        max_cells: Maximum cell threshold for filtering genes.
    """

    def __init__(
        self,
        min_counts: int | None = None,
        min_cells: int | None = None,
        max_counts: int | None = None,
        max_cells: int | None = None,
    ):
        """
        Initialize the FilterGenesTransform class.

        Args:
        ----
            min_counts: Minimum count threshold for filtering genes.
            min_cells: Minimum cell threshold for filtering genes.
            max_counts: Maximum count threshold for filtering genes.
            max_cells: Maximum cell threshold for filtering genes.
        """
        self.min_counts = min_counts
        self.min_cells = min_cells
        self.max_counts = max_counts
        self.max_cells = max_cells

    def __call__(self, adata: AnnData) -> Mapping[str, AnnData]:
        """
        Apply the filter genes transformation to the provided input data.

        Args:
        ----
            data:  The input AnnData object.

        Returns:
        -------
            AnnData: The transformed AnnData object.

        Raises:
        ------
            ValueError: If the input data is not an AnnData object.
            ValueError: If the transformation fails.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("Input data must be an AnnData object.")

        # Apply the filter genes transformation
        try:
            logger.info("in FilterGenes")
            logger.info(f"adata shape before = {adata.to_df().shape}")
            sc.pp.filter_genes(
                adata,
                min_counts=self.min_counts,
                min_cells=self.min_cells,
                max_counts=self.max_counts,
                max_cells=self.max_cells,
            )
            logger.info(f"adata shape after = {adata.to_df().shape}")
        except Exception as e:
            raise ValueError(
                "Error occurred during filter_genes transformation:", str(e)
            )

        # Return the transformed AnnData object
        return {"adata": adata}


class NormalizeTotalTransform(BaseSCTransform):
    """
    A class for applying the scanpy.pp.normalize_total function as a
    transformation.

    Attributes
    ----------
        target_sum: Target sum of counts per cell after normalization.
        exclude_highly_expressed: Exclude highly expressed genes from normalization.
        max_fraction: Maximum fraction of counts per cell to consider as highly expressed genes.
    """

    def __init__(
        self,
        target_sum: float | None = None,
        exclude_highly_expressed: bool = False,
        max_fraction: float = 0.05,
    ):
        """
        Initialize the NormalizeTotalTransform class.

        Args:
        ----
            target_sum: Target sum of counts per cell after normalization.
            exclude_highly_expressed: Exclude highly expressed genes from normalization.
            max_fraction: Maximum fraction of counts per cell to consider as highly expressed genes.
        """
        self.target_sum = target_sum
        self.exclude_highly_expressed = exclude_highly_expressed
        self.max_fraction = max_fraction

    def __call__(self, adata: AnnData) -> Mapping[str, AnnData]:
        """
        Apply the normalize total transformation to the provided AnnData
        object.

        Args:
        ----
            adata: The input AnnData object.

        Returns:
        -------
            AnnData: The transformed AnnData object.

        Raises:
        ------
            ValueError: If the input data is not an AnnData object.
            ValueError: If the transformation fails.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("Input data must be an AnnData object.")

        try:
            # Apply the normalize total transformation
            sc.pp.normalize_total(
                adata,
                target_sum=self.target_sum,
                exclude_highly_expressed=self.exclude_highly_expressed,
                max_fraction=self.max_fraction,
                inplace=True,
            )
        except Exception as e:
            raise ValueError(
                "Error occurred during normalize_total transformation:", str(e)
            )

        return {"adata": adata}


class ScaleCountsTransform(BaseSCTransform):
    """
    Scale all counts in anndata object by the same factor.

    Attributes
    ----------
        scale_factor (float): all counts will be scaled by this factor
    """

    def __init__(
        self,
        scale_factor: float | None = None,
    ):
        """
        Initialize the NormalizeTotalTransform class.

        Args:
        ----
            target_sum: Target sum of counts per cell after normalization.
            exclude_highly_expressed: Exclude highly expressed genes from normalization.
            max_fraction: Maximum fraction of counts per cell to consider as highly expressed genes.
        """
        self.scale_factor = scale_factor

    def __call__(self, adata: AnnData) -> Mapping[str, AnnData]:
        """
        Apply the scale_counts to the provided AnnData object.

        Args:
        ----
            adata: The input AnnData object.

        Returns:
        -------
            AnnData: The transformed AnnData object.

        Raises:
        ------
            ValueError: If the input data is not an AnnData object.
            ValueError: If the transformation fails.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("Input data must be an AnnData object.")

        try:
            adata.X *= self.scale_factor
        except Exception as e:
            raise ValueError(
                "Error occurred during scale_counts transformation:", str(e)
            )

        return {"adata": adata}


class LogTransform(BaseSCTransform):
    """
    Class for applying the scanpy.pp.log1p function as a transformation.

    Attributes
    ----------
        base: The base value for the logarithm. Default is natural logarithm (e).
        chunked: Whether to chunk the computation for large data. Default is False.
        chunk_size: Size of each chunk for chunked computation. Default is None.
        add_one: Add one to value before the log operation. False value allows computation log2 with log1p method. Default is True.
    """

    def __init__(
        self,
        base: float | None = None,
        chunked: bool | None = None,
        chunk_size: int | None = None,
        add_one: bool = True,
    ):
        """
        Initialize the LogTransform class.

        Args:
        ----
            base: The base value for the logarithm. Default is natural logarithm (e).
            chunked: Whether to chunk the computation for large data. Default is False.
            chunk_size: Size of each chunk for chunked computation. Default is None.
            add_one: Add one to value before the log operation aka log1p.
                If False, log1p is called but data is input as data-1 to neutralize the effect.
        """
        self.base = base
        self.chunked = chunked
        self.chunk_size = chunk_size
        self.add_one = add_one

    def __call__(self, adata: AnnData) -> Mapping[str, AnnData]:
        """
        Apply the log transformation to the provided AnnData object.

        Args:
        ----
            adata: The input AnnData object.

        Returns:
        -------
            AnnData: The transformed AnnData object.

        Raises:
        ------
            ValueError: If the input data is not an AnnData object.
            ValueError: If the transformation fails.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("Input data must be an AnnData object.")

        try:
            if not self.add_one:
                if isinstance(adata.X, csr_matrix):
                    adata.X = csr_matrix(csr_matrix.toarray(adata.X) - 0.999999)
                else:
                    adata.X = adata.X - 0.999999
            sc.pp.log1p(
                adata,
                base=self.base,
                chunked=self.chunked,
                chunk_size=self.chunk_size,
            )
            if not self.add_one:
                if isinstance(adata.X, csr_matrix):
                    adata.X = csr_matrix(
                        np.clip(a=csr_matrix.toarray(adata.X), a_min=0, a_max=None)
                    )
                else:
                    adata.X = np.clip(a=adata.X, a_min=0, a_max=None)

        except Exception as e:
            raise ValueError("Error occurred during log transformation:", str(e))

        return {"adata": adata}


class HighlyVariableGenesTransform(BaseSCTransform):
    """
    Class for applying the scanpy.pp.highly_variable_genes function as a
    transformation.

    Attributes:
    ----------
        n_top_genes: Number of top highly variable genes to select.
        min_disp: Minimum gene dispersion threshold.
        max_disp: Maximum gene dispersion threshold.
        min_mean: Minimum gene mean threshold.
        max_mean: Maximum gene mean threshold.
        span: Span parameter for the seurat flavor.
        n_bins: Number of bins for the seurat flavor.
        flavor: Flavor of the highly variable gene selection method.
        subset: Whether to subset the data based on highly variable genes.

    Returns:
    -------
        AnnData: The transformed AnnData object.

    Example:
    -------
        data = ...  # Your input AnnData object

        # Create an instance of the HighlyVariableGenesTransform class
        hvg_transform = HighlyVariableGenesTransform(n_top_genes=100, min_disp=0.2, max_mean=10)

        # Apply the transformation to the data
        transformed_data = hvg_transform(data)
    """

    def __init__(
        self,
        n_top_genes: int | None = None,
        min_disp: float = 0.5,
        max_disp: float = float("inf"),
        min_mean: float = 0.0125,
        max_mean: float = 3,
        span: float = 0.3,
        n_bins: int = 20,
        flavor: str = "seurat",
        subset: bool = True,
        batch_key: str | None = None,
    ):
        """
        Initialize the HighlyVariableGenesTransform class.

        Args:
        ----
            n_top_genes: Number of top highly variable genes to select.
            min_disp: Minimum gene dispersion threshold.
            max_disp: Maximum gene dispersion threshold.
            min_mean: Minimum gene mean threshold.
            max_mean: Maximum gene mean threshold.
            span: Span parameter for the seurat flavor.
            n_bins: Number of bins for the seurat flavor.
            flavor : {'seurat', 'seurat_v3', 'cell_ranger', 'pearson_residuals'}, optional
                The method used to identify highly variable genes.
                Each flavor assumes a specific form of preprocessing:

                - **'seurat'**
                Based on Seurat v2's variance stabilization approach.
                Use this flavor **only with log-transformed data** (e.g. after `sc.pp.log1p`), typically
                computed from library-size normalized counts (via `sc.pp.normalize_total`).
                Raw counts should *not* be passed directly.

                - **'seurat_v3'**
                Implements Seurat v3's mean-variance relationship model.
                Expects **raw count data** (unnormalized integer counts).
                The function internally normalizes and log-transforms the data before computing
                variance residuals.
                Do *not* pre-log or pre-normalize before calling.

                - **'cell_ranger'**
                Emulates the approach used by the Cell Ranger pipeline.
                Use **raw counts** as input (integer UMI counts).
                The function automatically applies total-count normalization and log transformation.

                - **'pearson_residuals'**
                Designed for **raw count data**, using Pearson residuals from a negative binomial model
                fitted per gene.
                Do *not* normalize or log-transform in advance, as the method models count variance
                explicitly.

            subset: Whether to subset the data based on highly variable genes.
            batch_key : str, optional
                Column in `adata.obs` that identifies batch membership (e.g. sequencing run, plate, donor, or perturbation group).
                When provided, highly variable genes are computed **separately within each batch**, then aggregated across batches.
                This can help control for technical variation but must be used carefully:

                - **Good idea:**
                When batches represent *technical replicates* (e.g. sequencing runs, plates, donors)
                and each contains a comparable mix of biological conditions.
                Batch-aware HVG selection helps avoid picking genes variable only due to batch effects.

                - **Bad idea:**
                When batches are *biologically distinct conditions*, such as different **perturbations**, **cell types**,
                or **time points**.
                In perturbation datasets where each batch corresponds to a unique treatment or perturbation,
                using `batch_key` can suppress truly condition-specific variability—resulting in under-selection
                of relevant response genes.
                In such cases, prefer running HVG selection on the **unbatched data** or **within each perturbation subset** separately.

        Notes:
        -----
        Choosing the correct flavor depends on the preprocessing stage of your AnnData object:
            - If you've already normalized and log-transformed your data → use `'seurat'`.
            - If you're starting from raw counts and want the modern default → use `'seurat_v3'`.
            - For Cell Ranger-style variance estimates → use `'cell_ranger'`.
            - For NB residual-based HVG selection → use `'pearson_residuals'`.
        """
        self.n_top_genes = n_top_genes
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.min_mean = min_mean
        self.max_mean = max_mean
        self.span = span
        self.n_bins = n_bins
        self.flavor = flavor
        self.subset = subset
        self.batch_key = batch_key

    def __call__(self, adata: AnnData) -> Mapping[str, AnnData]:
        """
        Apply the highly variable genes transformation to the provided
        AnnData object.

        Args:
        ----
            adata: The input AnnData object.

        Raises:
        ------
            ValueError: If the input data is not an AnnData object.
            ValueError: If the flavor is not supported.

        Returns:
        -------
            AnnData: The transformed AnnData object.
        """
        if not isinstance(adata, AnnData):
            raise ValueError("Input data must be an AnnData object.")

        try:
            # Apply the highly variable genes transformation
            logger.info("in HighlyVariableGenesTransform")
            logger.info(f"adata shape before = {adata.to_df().shape}")

            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=self.n_top_genes,
                min_disp=self.min_disp,
                max_disp=self.max_disp,
                min_mean=self.min_mean,
                max_mean=self.max_mean,
                span=self.span,
                n_bins=self.n_bins,
                flavor=self.flavor,
                subset=self.subset,
                check_values=True,
                batch_key=self.batch_key,
            )
            logger.info(f"adata shape after = {adata.to_df().shape}")

        except Exception as e:
            raise ValueError(
                "Error occurred during highly_variable_genes transformation:", str(e)
            )
        return {"adata": adata}


class MedianNormalizeTransform(BaseSCTransform):
    """
    A class for performing median normalization on data.


    Args:
    ----
        file | dict: The file or dictionary containing the median values for each gene.


    Returns:
    -------
        AnnData: The transformed AnnData object.
    """

    def __init__(
        self, path_to_median_data: str | None = None, median_dict: dict | None = None
    ):
        """
        Initialize the MedianNormalizeTransform class.


        Args:
        ----
        path_to_median_data: The path to the file containing the median values for each gene.
        median_dict: The dictionary containing the median values for each gene.


        Returns:
        -------
            AnnData: The transformed AnnData object.


        Raises:
        ------
            ValueError: If the path to the file is not a string and the median data is not a dictionary.
            ValueError: If both the path to the file and the median data are provided.
        """
        self.median_dict = median_dict
        self.path_to_median_data = path_to_median_data

        if self.path_to_median_data is None and self.median_dict is None:
            raise ValueError("File must be a string or a dictionary.")

        if self.path_to_median_data is not None and self.median_dict is not None:
            raise ValueError(
                "Only one of path_to_median_data or median_data must be provided."
            )
        if self.path_to_median_data is not None:
            with open(self.path_to_median_data, "rb") as f:
                self.median_dict = pickle.load(f)

    def __call__(self, adata: AnnData) -> Mapping[str, AnnData]:
        """
        Apply the median normalization transformation to the input AnnData object.


        Args:
        ----
            adata (AnnData): Input AnnData object to be transformed.


        Returns:
        -------
            AnnData: Transformed AnnData object with median normalized data.


        Raises:
        ------
            ValueError: If the input data is not a 2D array.
        """
        logger.info(
            "Median normalization started {}".format(
                datetime.datetime.now().strftime("%H:%M:%S")
            )
        )
        if adata.X.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        logger.info("Median normalizing data ...")
        assert self.median_dict is not None
        sparse_matrix = adata.X
        var_names = adata.var_names
        median_data = []
        for i in range(adata.shape[0]):
            row = sparse_matrix.data[
                sparse_matrix.indptr[i] : sparse_matrix.indptr[i + 1]
            ]
            indices = sparse_matrix.indices[
                sparse_matrix.indptr[i] : sparse_matrix.indptr[i + 1]
            ]
            median_values_row = [
                self.median_dict[var_names[index]] for index in indices
            ]
            median_data.extend(row / median_values_row)

        adata.X = csr_matrix(
            (median_data, sparse_matrix.indices, sparse_matrix.indptr),
            shape=sparse_matrix.shape,
        )
        logger.info(
            "Median normalization done {}".format(
                datetime.datetime.now().strftime("%H:%M:%S")
            )
        )
        return {"adata": adata}


class BinTransform(BaseSCTransform):
    """
    A class for performing binning on data.

    Args:
    ----
        num_bins (int): Number of bins for the binning transformation.
        result_bin_key (str): The key to use for storing the binned data.

    Returns:
    -------
        AnnData: The transformed AnnData object.
    """

    def __init__(
        self,
        num_bins: int | None = None,
        binning_method: str = "percentile",
    ):
        """
        Initialize the BinTransform class.

        Args:
        ----
            num_bins: Number of bins for the binning transformation.
            binning_method (str): method used to bin the expression levels.
                Supported values are:
                  - "percentile": calculates bin edges based on percentiles of the
                    expression data. This follows the method used in the scGPT paper.
                  - "int_cast": casts the float values of the expression to integers,
                    using those values as the bin numbers. The cast is after `ceil` so
                    that no data points are cast to zero. This is usually done after a
                    log1p transform. This follows the method used in the scBERT paper.
        """
        self.num_bins = num_bins
        self.binning_method = binning_method

    def _digitize(self, x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
        """
        Digitize the data into bins. This method spreads data uniformly when bins
        have same values.

        Args:
        ----
        x (:class:`np.ndarray`):
            The data to digitize.
        bins (:class:`np.ndarray`):
            The bins to use for digitization, in increasing order.
        side (:class:`str`, optional):
            The side to use for digitization. If "one", the left side is used. If
            "both", the left and right side are used. Default to "one".

        Returns:
        -------
        :class:`np.ndarray`:
            The digitized data.
        """
        left_digits = np.digitize(x, bins)
        if side == "one":
            return left_digits

        right_digits = np.digitize(x, bins, right=True)

        rands = np.random.rand(len(x))  # uniform random numbers

        digits = rands * (right_digits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits

    def __call__(self, adata: AnnData) -> Mapping[str, AnnData]:
        """
        Apply the binning transformation to the input AnnData object.

        Args:
        ----
            adata (AnnData): Input AnnData object to be transformed.

        Returns:
        -------
            AnnData: Transformed AnnData object with binned data.

        Raises:
        ------
            ValueError: If the input data is not a 2D array.
        """
        logger.info(
            "Binning started {}".format(datetime.datetime.now().strftime("%H:%M:%S"))
        )
        binned_data = None
        if adata.X.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        logger.info("Binning data ...")

        if isinstance(adata.X, csr_matrix):
            sparse_matrix = adata.X
        elif isinstance(adata.X, np.ndarray):
            sparse_matrix = csr_matrix(adata.X)
        else:
            sparse_matrix = adata.X.tocsr()

        if self.binning_method == "int_cast":
            binned_data = np.ceil(sparse_matrix.data).astype(np.int64)
            if self.num_bins is not None:
                binned_data = np.clip(
                    a=binned_data, a_min=None, a_max=self.num_bins - 1
                )

        elif self.binning_method == "percentile":
            if self.num_bins is None:
                raise ValueError(
                    "The number of bins must be specified when using the percentile binning method."
                )
            percentiles = np.linspace(0, 100, self.num_bins + 1)
            binned_data = []

            for i in range(adata.shape[0]):
                row = sparse_matrix.data[
                    sparse_matrix.indptr[i] : sparse_matrix.indptr[i + 1]
                ]
                bin_edges = np.percentile(row, percentiles)
                binned_values = np.clip(
                    np.digitize(row, bin_edges), 1, len(bin_edges) - 1
                )
                binned_data.extend(binned_values)

        elif self.binning_method == "value_binning":
            if self.num_bins is None:
                raise ValueError(
                    "The number of bins must be specified when using the value binning method."
                )
            binned_data = []

            for i in range(adata.shape[0]):
                non_zero_row = sparse_matrix.data[
                    sparse_matrix.indptr[i] : sparse_matrix.indptr[i + 1]
                ]
                bins = np.quantile(non_zero_row, np.linspace(0, 1, self.num_bins))
                binned_values = self._digitize(non_zero_row, bins)
                binned_data.extend(binned_values)
        else:
            raise ValueError(f"Unknown binning method: {self.binning_method}")
        values, counts = np.unique(binned_data, return_counts=True)
        zip_iterator = zip(values, counts)
        dict_counts = dict(zip_iterator)
        logger.info(f"Number of bins: {dict_counts}")

        adata.X = csr_matrix(
            (binned_data, sparse_matrix.indices, sparse_matrix.indptr),
            shape=sparse_matrix.shape,
        )

        logger.info(
            "Binning done {}".format(datetime.datetime.now().strftime("%H:%M:%S"))
        )
        return {"adata": adata}


def make_transform(
    transform_name: str, transform_args: Mapping[str, Any]
) -> BaseSCTransform:
    """
    Factory function for creating SCTransform objects.

    Args:
    ----
        transform_name: The name of the transform to create.
        transform_args: The arguments to pass to the transform.

    Returns:
    -------
        transform: The created transform object.
    """
    factory_function = lookup_name(transform_name)
    return factory_function(**transform_args)
