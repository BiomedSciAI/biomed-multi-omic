import argparse
from pathlib import Path

import pandas as pd
import pyBigWig
from tqdm import tqdm


def get_gene_promoters(
    gtf_path: str, promoter_up: int, promoter_down: int
) -> pd.DataFrame:
    """
    Parses a GTF file to get the promoter region for each gene.
    NOTE: This uses a simplified method of taking the first transcript found for
    each gene name. It's a good starting point for verification.
    """
    print(f"1. Reading gene annotations from {gtf_path}...")
    gtf_cols = [
        "chrom",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attrs",
    ]
    gtf_df = pd.read_csv(
        gtf_path, sep="\t", comment="#", names=gtf_cols, low_memory=False
    )
    transcripts = gtf_df[gtf_df["feature"] == "transcript"].copy()
    transcripts["gene_name"] = transcripts["attrs"].str.extract('gene_name "([^"]+)"')
    transcripts = transcripts.dropna(subset=["gene_name"])
    canonical = transcripts.groupby("gene_name").first().reset_index()
    canonical["tss"] = canonical.apply(
        lambda row: row["start"] if row["strand"] == "+" else row["end"], axis=1
    )
    canonical["promoter_start"] = (canonical["tss"] - promoter_up).clip(lower=0)
    canonical["promoter_end"] = canonical["tss"] + promoter_down
    print(f"   Found {len(canonical)} unique genes.")
    return canonical[["gene_name", "chrom", "promoter_start", "promoter_end"]]


def calculate_scores(
    genes_df: pd.DataFrame, bigwig_path: str, feature_name: str
) -> pd.DataFrame:
    """
    Calculates the mean signal for each gene's promoter using the bigWig file.
    This version includes a fix to handle intervals near chromosome ends.
    """
    print(f"2. Opening signal file {bigwig_path}...")
    try:
        bw = pyBigWig.open(bigwig_path)
    except RuntimeError as e:
        print(f"Error opening bigWig file: {e}")
        return pd.DataFrame()

    # Get the lengths of all chromosomes from the bigWig file
    chrom_lengths = bw.chroms()

    results = []
    print("3. Calculating mean signal for each gene promoter...")

    for _, gene in tqdm(genes_df.iterrows(), total=len(genes_df)):
        chrom = gene["chrom"]
        start = int(gene["promoter_start"])
        end = int(gene["promoter_end"])

        # Check if the chromosome from the GTF exists in the bigWig file
        if chrom in chrom_lengths:
            # Cap the end coordinate at the chromosome's actual length
            chrom_max_len = chrom_lengths[chrom]
            end = min(end, chrom_max_len)

            # Ensure the interval is still valid (start < end) after clipping
            if start >= end:
                continue  # Skip invalid intervals

            mean_signal = bw.stats(chrom, start, end, type="mean")[0]

            results.append(
                {
                    "gene_symbol": gene["gene_name"],
                    feature_name: mean_signal if mean_signal is not None else 0.0,
                }
            )

    bw.close()
    return pd.DataFrame(results)


def write_feature_to_datastore(
    feature_df: pd.DataFrame,
    datastore_path: str | Path,
    feature_name: str,
    biosample_name: str,
    gene_col: str = "gene_symbol",
    context_col: str = "biosample_name",
):
    """
    Adds or updates a feature column in a global datastore in wide format.
    Keeps one row per (gene_symbol, biosample_name) pair and one column per feature.
    """
    datastore_path = Path(datastore_path)

    # Normalize input
    feature_df = feature_df[[gene_col, feature_name]].copy()
    feature_df[context_col] = biosample_name

    if datastore_path.exists():
        ds = pd.read_parquet(datastore_path)

        # Ensure consistent columns
        if feature_name not in ds.columns:
            ds[feature_name] = pd.NA

        # Merge by (gene, context) keys, preferring new data
        ds = ds.set_index([gene_col, context_col])
        feature_df = feature_df.set_index([gene_col, context_col])

        # Upsert: overwrite or add missing rows
        ds.update(feature_df)
        ds = pd.concat([ds, feature_df[~feature_df.index.isin(ds.index)]])
        ds = ds.reset_index()
    else:
        ds = feature_df

    ds.to_parquet(datastore_path, index=False)
    print(
        f"✅ Updated datastore with feature '{feature_name}' for context '{biosample_name}'"
    )


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Calculate gene-level scores and add to datastore."
    )
    parser.add_argument(
        "--bigwig", required=True, help="Path to the input bigWig signal file."
    )
    parser.add_argument(
        "--gtf",
        required=True,
        help="Path to the gene annotation GTF file (gzipped is ok).",
    )
    parser.add_argument(
        "--datastore", required=True, help="Path to the global datastore parquet file."
    )
    parser.add_argument(
        "--biosample_name",
        required=True,
        help="Biosample name (e.g., name of the cell_line or cell_type). This will be used for looking up the correct epigenetic data during training.",
    )
    parser.add_argument(
        "--feature_name",
        default="h3k4me1_promoter_signal",
        help="Feature column name to store. This will be the field name during training.",
    )
    parser.add_argument(
        "--promoter_up",
        type=int,
        default=1000,
        help="Bases upstream of TSS for promoter window.",
    )
    parser.add_argument(
        "--promoter_down",
        type=int,
        default=1000,
        help="Bases downstream of TSS for promoter window.",
    )
    args = parser.parse_args()

    promoters = get_gene_promoters(args.gtf, args.promoter_up, args.promoter_down)

    if not promoters.empty:
        scores_df = calculate_scores(promoters, args.bigwig, args.feature_name)

        if not scores_df.empty:
            write_feature_to_datastore(
                scores_df,
                datastore_path=args.datastore,
                feature_name=args.feature_name,
                biosample_name=args.biosample_name,
            )


if __name__ == "__main__":
    main()
