# Epigenetic data

Gene expression is regulated in part by epigenetics, modifications to the DNA molecules that do not affect the genetic code itself but do affect how it is utilized.

Though this data is not nearly as widely available as scRNA, for cell lines a robust set of measurements have been compiled, and nearly all perturbation experiments are conducted on cell lines.

## Data source

We have focused on the [ENCODE reference genomes](https://www.encodeproject.org/reference-epigenome-matrix/?type=Experiment&control_type!=*&related_series.@type=ReferenceEpigenome&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&status=released) for cell lines which offer a curated set of measurements, eg [ENCSR612NLL](https://www.encodeproject.org/reference-epigenomes/ENCSR612NLL/) for K562 and [ENCSR820QMS](https://www.encodeproject.org/reference-epigenomes/ENCSR820QMS/) for H1.

We have selected a broad selection of potentially relevant epigenetic markers to experiment with for training.

## Data collection

Because of the illegible study names and multiple versions, we have added a simple data collection process that balances
provenance tracking, validation and simplicity.

1. Create a list of files to download based on the reference genome pages. This must be done manually, or with permitted AI tools. The final list should look like [this](./encode_file_list_k562.tsv):

    ```tsv
    #cell_line	measurement_name	experiment_id	file_id
    K562	H3K27ac	ENCSR000AKP	ENCFF465GBD
    K562	H3K4me1	ENCSR000AKS	ENCFF457URZ
    K562	H3K9me3	ENCSR000APE	ENCFF632NQA
    K562	H3K36me3	ENCSR000DWB	ENCFF633OZC
    K562	H3K27me3	ENCSR000EWB	ENCFF847BFA
    K562	H3K4me3	ENCSR668LDD	ENCFF405ZDL
    K562	methylation_plus	ENCSR765JPC	ENCFF459XNY
    K562	methylation_minus	ENCSR765JPC	ENCFF430PNX
    K562	Pol2S5	ENCSR000BKR	ENCFF677XKP
    K562	Pol2	ENCSR388QZF	ENCFF914WIS
    K562	EP300	ENCSR000EGE	ENCFF325DSL
    K562	CTCF	ENCSR000EGM	ENCFF336UPT
    K562	DNase	ENCSR000EOT	ENCFF414OGC
    K562	ATAC	ENCSR868FGK	ENCFF357GNC
    K562	smallRNA_plus	ENCSR000AES	ENCFF259NAV
    K562	smallRNA_minus	ENCSR000AES	ENCFF967BLD
    ```

2. A canonical list of modalities, experiment and file names should be downloaded from ENCODE, such as `https://www.encodeproject.org/report/?type=Experiment&control_type%21=%2A&status=released&perturbed=false&files.file_type=bigWig&biosample_ontology.term_name=K562&limit=all` This is a reproducible, verifiable way to ensure that the downloaded files in fact reflect the desired measurements and cell lines.
3. Use the script [encode_experiment_file_validation](./encode_experiment_file_validation.sh) to verify that your file list is consistent with the canonical list.
4. Use the script [encode_download_bigwig_list](./encode_download_bigwig_list.sh) to download and rename the files into a legible directory structure.

### Methylation processing

Methylation data is measured separately for the plus and minus strands. For the purposes of our study, we want a single methylation measurement, so we combine the plus and minus strands using the script [combine_plus_minus_methylation](./combine_plus_minus_methylation.sh).

Because this operation combines forward and backward chromosome reading, it requires the chromosome sizes, which can be obtained here: [hg38.chrom.sizes](https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes)

## Calculating gene-level epigenetics to use for training

Epigenetic measurements are collected at the base pair level, not the gene level. Broadly speaking the epigenetics locations affecting a gene's expression are in the vicinity of its promoter (near where transcription of the gene begins), or enhancer(s) (distant regulatory sites).
At this time, we support promoter measurements only.

The downloaded bigwig files are scanned by gene, by identifying the transcription start site (TSS) from the canonical gene annotation, which must be downloaded (eg [gencode.v38.annotation.gtf.gz](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz)), and averaging the epigenetic contributions within a certain number of base pairs before and after the TSS (default=1000).

Finally, this produces a table of gene symbols and floating point numbers for each epigenetic measurement. The script [load_bigwig.py](./load_bigwig.py) adds a single file to this table, and assuming the files have been downloaded into the directory structure produced by the above scripts the helper [load_to_datastore.sh](./load_to_datastore.sh) adds all of the bigwigs from a root directory structure with the correct names.

The final result of this process will be a datastore, a parquet dataframe with columns `gene_symbol`, `biosample_name` referring to the cell line and additional columns for all of the available epigenetic fields.

## Using epigenetic fields during training

To access epigenetic fields during training, the dataset must have a column in `obs` that has values matching the `biosample_name`, ie a column called `"cell_line"` with value `"K562"`.

The only modifications to the training yaml to access these features are:

```yaml
fields:
  # regular fields, such as "genes", "expressions", "perturbations"
  - _target_: bmfm_targets.config.FieldInfo
    field_name: methylation # must have a corresponding column in the datastore
    tokenization_strategy: continuous_value_encoder
    encoder_kwargs:
      kind: scale_adapt
      n_sin_basis: 48
      basis_scale: 1.5
      trainable: true
      zero_as_special_token: true
    datastore_config:
      _target_: bmfm_targets.config.tokenization_config.DatastoreConfig
      path: /data/omics/epigenome/datastore.parquet # must point to the datastore produced above
label_columns:
  - _target_: bmfm_targets.config.LabelColumnInfo
    label_column_name: cell_line
    is_bio_context_for_datastore: true # this flag can be used for only one label column
```
