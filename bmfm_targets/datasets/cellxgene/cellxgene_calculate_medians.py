#!/usr/bin/env python

import argparse
import json
from itertools import islice
from pathlib import Path

import cellxgene_census
from tqdm import tqdm


def batched(iterable, n):
    """From itertools, for comp. with old python versions."""
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def build_gene_query_filter(genes):
    gene_str = ",".join(["'" + i + "'" for i in genes])
    value_filter = f"feature_id in [{gene_str}]"
    return value_filter


def _calculate_median(data):
    """Faster function than statistics.median."""
    data.sort()
    n = len(data)
    if n % 2 == 1:
        return data[n // 2]
    else:
        i = n // 2
        return (data[i - 1] + data[i]) / 2


class SOMA:
    def __init__(
        self,
        uri,
        census_version="2023-12-15",
        value_filter="is_primary_data==True",
        gene_chunk_size=500,
    ):
        self.census = cellxgene_census.open_soma(uri=uri, census_version=census_version)
        self.value_filter = value_filter
        self.gene_chunk_size = gene_chunk_size

    def _caclulate_gene_medians(self, value_filter):
        adata = cellxgene_census.get_anndata(
            census=self.census,
            organism="homo sapiens",
            X_name="normalized",
            var_value_filter=value_filter,
            obs_value_filter=self.value_filter,
            column_names={"var": ["feature_name", "feature_id"], "obs": []},
        )
        matrix = adata.X.transpose().tocsr()
        indptr = matrix.indptr
        medians = []
        for index in range(matrix.shape[0]):
            gene_data = matrix.data[indptr[index] : indptr[index + 1]]
            if len(gene_data) == 0:
                medians.append(None)
            else:
                medians.append(_calculate_median(gene_data).item())
        medians = {
            f"{id}, {name}": median
            for id, name, median in zip(
                adata.var["feature_id"].values,
                adata.var["feature_name"].values,
                medians,
            )
        }
        return medians

    def calculate_medians(self):
        soma_experiment = self.census["census_data"]["homo_sapiens"]
        genes = (
            soma_experiment.ms["RNA"]
            .var.read(column_names=["feature_id"])
            .concat()
            .to_pandas()
            .squeeze()
            .values
        )
        medians = {}
        for gene_batch in tqdm(batched(genes, n=self.gene_chunk_size)):
            value_filter = build_gene_query_filter(gene_batch)
            medians |= self._caclulate_gene_medians(value_filter)
        return medians


def path_type_not_exists(name):
    path = Path(name)
    if not path.exists():
        raise FileNotFoundError(f"The directory {name} does not exists.")
    return name


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--inputdir",
        type=path_type_not_exists,
        default="/dccstor/bmfm-targets/data/omics/transcriptome/scRNA/pretrain/cellxgene/soma-2023-12-15",
        help="Input directory with CELLxGENE dataset.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="medians.json",
        help="File name for output with calculated medians.",
    )

    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        default="is_primary_data==True",
        help="Value filter.",
    )

    parser.add_argument(
        "-c", "--chunk_size", help="Gene chunk size.", type=int, default=500
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    soma = SOMA(
        args.inputdir, value_filter=args.filter, gene_chunk_size=args.chunk_size
    )
    medians = soma.calculate_medians()
    with open(args.output, "w") as file:
        json.dump(medians, file)
