import numpy as np
import pandas as pd

from bmfm_targets import config


class EpigeneticDatastore:
    """
    Provides efficient access to feature values (e.g., epigenetic marks)
    for specific biosamples and genes from a Parquet-based datastore.
    """

    def __init__(
        self,
        path: str,
        gene_column: str = "gene_symbol",
        bio_context_column: str = "biosample_name",
        nan_fill_value: int | float = 0,
        log1p_transform: bool = False,
    ):
        """
        Initialize the datastore by loading a Parquet file and indexing
        it by biosample for fast per-gene lookup.

        Parameters
        ----------
        path : str
            Path to the datastore Parquet file.
        gene_column : str, optional
            Name of the column identifying genes (default: "gene_symbol").
        bio_context_column : str, optional
            Name of the column identifying the biological context or sample
            (default: "biosample_name").
        nan_fill_value : int or float, optional
            Value used to replace NaNs in the datastore (default: 0).
        log1p_transform : bool, optional
            Whether to apply log1p transformation to values on retrieval (default: False).
        """
        self.df = pd.read_parquet(path).fillna(nan_fill_value)

        self.gene_column = gene_column
        self.bio_context_column = bio_context_column
        self.log1p_transform = log1p_transform
        # Build dict for fast lookup: biosample → {gene → feature_dict}
        self.bio_context_to_value_map = {
            ctx: sub.set_index(gene_column).to_dict(orient="index")
            for ctx, sub in self.df.groupby(bio_context_column)
        }

    def get_values(self, biosample_name: str, genes: list[str], feature: str):
        """
        Retrieve feature values for a given biosample and list of genes.

        Parameters
        ----------
        biosample_name : str
            Name of the biosample (must exist in the datastore).
        genes : list of str
            List of gene identifiers to retrieve.
        feature : str
            Name of the feature (e.g., histone mark, methylation level).

        Returns
        -------
        list of float
            Values aligned to `genes`. Missing values are replaced with 0.
            If `log1p_transform` is enabled, values are returned as log1p(x).
        """
        gene2value = self.bio_context_to_value_map[biosample_name]
        output = [gene2value.get(g, {}).get(feature, 0.0) for g in genes]
        if self.log1p_transform:
            return np.log1p(output).tolist()
        return output

    @classmethod
    def from_config(cls, field_info: config.FieldInfo):
        """
        Construct an `EpigeneticDatastore` from a `FieldInfo` object
        containing a nested `DatastoreConfig`.

        Parameters
        ----------
        field_info : config.FieldInfo
            Field definition containing a `datastore_config` attribute.

        Returns
        -------
        EpigeneticDatastore
            Instantiated datastore configured according to the field's
            associated `DatastoreConfig`.

        Raises
        ------
        ValueError
            If the provided `field_info` lacks a `datastore_config`.
        """
        if field_info.datastore_config is None:
            raise ValueError(
                f"Field '{field_info.field_name}' has no datastore config."
            )
        cfg = field_info.datastore_config
        return cls(
            path=cfg.path,
            gene_column=cfg.token_index_column,
            bio_context_column=cfg.bio_context_column_in_datastore,
            log1p_transform=cfg.log1p_transform,
        )
