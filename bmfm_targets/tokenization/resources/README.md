# Tokenization resources
#### hgnc_complete_set_2025-08-23.tsv

Includes the HGNC gene list as downloaded from <https://storage.googleapis.com/public-download-files/hgnc/archive/archive/monthly/tsv/hgnc_complete_set_2024-08-23.tsv>


###
GSE70138_Broad_LINCS_gene_info_2017-03-06.txt
Holds the L1000 gene list as downloaded from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138

### load_bigwig.py

For loading epigenomic signal data (e.g., histone marks, chromatin accessibility, or methylation) stored in a BigWig file into gene-level summaries based on promoters, and inserting it into a datastore for access as a field during training.

```bash
 python ~/bmfm-targets/bmfm_targets/tokenization/resources/load_bigwig.py \\
--bigwig ~/Downloads/ENCFF457URZ.bigWig \\
--gtf ~/bmfm-targets/bmfm_targets/tokenization/resources/gencode.v38.annotation.gtf.gz \\
--datastore ~/bmfm-targets/data/epigenetics.parquet \\
--biosample_name K562 \\
--feature_name h3k4me1_promoter_signal\\
```
For this example, I wanted to get h3k4me1 data for K562 so I downloaded ENCFF457URZ.bigWig from https://www.encodeproject.org/files/ENCFF457URZ/
and the v38 assembly from https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz

The script computes the mean signal intensity over each gene’s promoter region and inserts the resulting gene-level values into a shared Parquet file.
This allows us to accumulate multiple epigenetic annotations (e.g., histone marks, chromatin accessibility, methylation) across multiple cell lines and cell types and access them in a unified way during training.
