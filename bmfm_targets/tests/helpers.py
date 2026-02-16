import os
import random
from collections import defaultdict
from pathlib import Path
from subprocess import call

import anndata as ad
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from scanpy import read_h5ad
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

from bmfm_targets import config
from bmfm_targets.config import LabelColumnInfo
from bmfm_targets.datasets import (
    DatasetTransformer,
    base_rna_dataset,
)
from bmfm_targets.datasets.datasets_utils import (
    get_split_column,
)
from bmfm_targets.tests import helpers
from bmfm_targets.tokenization import (
    MultiFieldInstance,
    MultiFieldTokenizer,
    get_snp2vec_tokenizer,
)
from bmfm_targets.training.losses import CrossEntropyObjective, FieldSource, LossTask

FINETUNE_ROOT = Path(__file__).parent / "resources" / "finetune"
PRETRAIN_ROOT = Path(__file__).parent / "resources" / "pretrain"


class TorchListDataset(Dataset):
    def __init__(self, list_data: list):
        self.list_data = list_data

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, index):
        return self.list_data[index]


def generate_dataset(
    size,
    min_seq_len=1000,
    max_seq_len=5000,
    seed=None,
    token_fields=None,
    tokenizer=None,
    scalar_valued_fields=None,
    zero_fraction: float | None = None,
    label_dict: dict[str, dict[str, int]] | None = None,
    is_rna_seq: bool = False,
):
    if token_fields is None:
        token_fields = ["genes", "expressions"]
    if tokenizer is None:
        tokenizer = load_test_tokenizer()
    if scalar_valued_fields is None:
        scalar_valued_fields = ["expressions"]
    field_tokens = {}

    for field in tokenizer.tokenizers.keys():
        if field not in token_fields:
            continue
        vocab = tokenizer.get_field_tokenizer(field).get_vocab()
        field_tokens[field] = [
            token for token in vocab.keys() if token not in tokenizer.all_special_tokens
        ]
    dataset = []
    if seed is not None:
        random.seed(seed)
    for i in range(size):
        seq_len = random.randint(min_seq_len, max_seq_len)
        data = {}
        for field in token_fields:
            if field == "genes" and is_rna_seq:
                data[field] = random.sample(field_tokens[field], seq_len)
            else:
                data[field] = random.choices(field_tokens[field], k=seq_len)
        for field in scalar_valued_fields:
            data[field] = [*np.random.lognormal(mean=0, sigma=1, size=seq_len)]
            if zero_fraction is not None:
                nonzero_mask = np.random.random(size=seq_len) > zero_fraction
                data[field] = [d if z else 0 for d, z in zip(data[field], nonzero_mask)]
        metadata = {"cell_name": f"dummy_{i}"}
        if label_dict is not None:
            for label_column_name, sublabel_dict in label_dict.items():
                metadata[label_column_name] = random.choice(list(sublabel_dict))
        dataset.append(MultiFieldInstance(data=data, metadata=metadata))
    return TorchListDataset(dataset)


def generate_sequence_labeling_perturbation_dataset(
    size,
    fields: list[config.FieldInfo],
    min_seq_len=1000,
    max_seq_len=5000,
    seed=None,
):
    tokenizer = load_test_tokenizer()
    dataset = []
    if seed is not None:
        random.seed(seed)
    for i in range(size):
        num_tokens = random.randint(min_seq_len, max_seq_len)
        data = {}
        for field in fields:
            vocab = tokenizer.get_field_tokenizer(field.field_name).get_vocab()
            field_tokens = list(vocab.keys())

            field_tokens = [
                float(token) if "expressions" in field.field_name else token
                for token in field_tokens
                if token not in tokenizer.all_special_tokens
            ]
            tokens = random.choices(field_tokens, k=num_tokens)
            data[field.field_name] = tokens

        metadata = {"cell_name": f"dummy_{i}"}
        dataset.append(MultiFieldInstance(data=data, metadata=metadata))
    return TorchListDataset(dataset)


def generate_hic_dataset(
    size,
    min_seq_len=100,
    max_seq_len=500,
    seed=None,
    token_fields=None,
    tokenizer=None,
):
    if token_fields is None:
        token_fields = ["dna_chunks"]
    if tokenizer is None:
        tokenizer = get_snp2vec_tokenizer()
    field_tokens = {}
    for field in tokenizer.tokenizers.keys():
        if field not in token_fields:
            continue
        vocab = tokenizer.get_field_tokenizer(field).get_vocab()
        field_tokens[field] = [
            token for token in vocab.keys() if token not in tokenizer.all_special_tokens
        ]
    dataset = []
    if seed is not None:
        random.seed(seed)
    for _ in range(size):
        seq_len = random.randint(min_seq_len, max_seq_len)
        data = {}
        for field in token_fields:
            data[field] = random.choices(field_tokens[field], k=seq_len)
        metadata = {"hic_contact": random.uniform(0, 5)}
        mfi1 = MultiFieldInstance(data=data, metadata=metadata)
        data = {}
        for field in token_fields:
            data[field] = random.choices(field_tokens[field], k=seq_len)
        metadata = {"hic_contact": random.uniform(0, 5)}
        mfi2 = MultiFieldInstance(data=data, metadata=metadata)
        dataset.append((mfi1, mfi2))
    return TorchListDataset(dataset)


def load_test_tokenizer() -> MultiFieldTokenizer:
    # Load the tokenizer
    tokenizer_root = Path(__file__).parent / "resources"
    return MultiFieldTokenizer(name_or_path=tokenizer_root)


def load_test_tokenizer_old() -> MultiFieldTokenizer:
    # Load the tokenizer
    vocab_path = Path(__file__).parent / "test_vocab"
    return MultiFieldTokenizer.from_pretrained(vocab_path)


def initialize_litdata():
    helpers.LitDataPaths.cache.mkdir(parents=True, exist_ok=True)
    helpers.LitDataPaths.data_cache.mkdir(parents=True, exist_ok=True)
    os.environ["DATA_OPTIMIZER_CACHE_FOLDER"] = str(helpers.LitDataPaths.cache)
    os.environ["DATA_OPTIMIZER_DATA_CACHE_FOLDER"] = str(
        helpers.LitDataPaths.data_cache
    )


TEST_GTF_PATH = Path(__file__).parent / "resources" / "GRCh38_latest_genomic_subset.gtf"
DATASTORE_PATH = Path(__file__).parent / "resources" / "epigenetics.parquet"


class Zheng68kPaths:
    root = FINETUNE_ROOT / "zheng68k"
    processed = root / "processed.h5ad"
    raw_counts_name = "raw_counts"
    processed_raw_counts = root / (raw_counts_name + ".h5ad")
    label_dict_path = root / "zheng68k_labels.json"


class AnnCollectionDatasetPaths:
    dataset = PRETRAIN_ROOT / "cellxgene" / "h5ad"
    root = PRETRAIN_ROOT / "AnnCollectionDataset"


class MultipleSclerosisPaths:
    root = FINETUNE_ROOT / "multiple_sclerosis"
    processed = root / "processed.h5ad"
    label_dict_path = (
        FINETUNE_ROOT / "multiple_sclerosis" / "multiple_sclerosis_labels.json"
    )


class HumanCellAtlasPaths:
    root = FINETUNE_ROOT / "humancellatlas"
    processed = root / "processed.h5ad"
    label_dict_path = FINETUNE_ROOT / "humancellatlas" / "humancellatlas_labels.json"


class MyeloidPaths:
    root = FINETUNE_ROOT / "myeloid"
    processed = root / "processed.h5ad"
    label_dict_path = FINETUNE_ROOT / "myeloid" / "myeloid_labels.json"


class hBonesPaths:
    root = FINETUNE_ROOT / "hBones"


class scIBDPaths:
    root = FINETUNE_ROOT / "scIBD"
    processed = root / "processed.h5ad"
    label_dict_path = FINETUNE_ROOT / "scIBD" / "scIBD_labels.json"


class scIBD300kPaths:
    root = FINETUNE_ROOT / "scIBD300k"
    processed = root / "processed.h5ad"
    processed_ac = root / "processed_ac.h5ad"
    label_dict_path = FINETUNE_ROOT / "scIBD300k" / "scIBD300k_labels.json"


class SciPlex3Paths:
    root = FINETUNE_ROOT / "sciplex3"


class TILPaths:
    root = FINETUNE_ROOT / "TIL"
    processed = root / "processed.h5ad"
    label_dict_path = FINETUNE_ROOT / "TIL" / "TIL_labels.json"


class SCP1884Paths:
    root = FINETUNE_ROOT / "SCP1884"
    processed = root / "processed.h5ad"
    label_dict_path = FINETUNE_ROOT / "SCP1884" / "scp1884_labels.json"


class CCFIBDPaths:
    root = Path("/dataset/stappenbeck/test/Healthy_ALI_Day7")
    processed = root / "processed.h5ad"
    label_dict_path = root / "ccfibd_labels.json"


class ChangEpiPaths:
    root = FINETUNE_ROOT / "changepi"
    processed = root / "processed.h5ad"
    label_dict_path = root / "ChangEpi_all_labels.json"


class ChangAllPaths:
    root = FINETUNE_ROOT / "changall"
    processed = root / "processed.h5ad"
    processed_ac = root / "processed_ac.h5ad"
    label_dict_path = root / "ChangAll_all_labels.json"


class ChangPouchitisPaths:
    root = FINETUNE_ROOT / "changpouchitis"
    processed = root / "processed.h5ad"
    label_dict_path = root / "ChangPouchitis_all_labels.json"


class BulkrnaIBDPaths:
    root = FINETUNE_ROOT / "bulkrnaIBD"
    processed = root / "processed.h5ad"
    label_dict_path = FINETUNE_ROOT / "bulkrnaIBD" / "bulkrnaIBD_labels.json"


class PanglaoPaths:
    root = PRETRAIN_ROOT / "panglao"
    all_metadata = PRETRAIN_ROOT / "panglao" / "metadata" / "metadata.csv"
    test_metadata = PRETRAIN_ROOT / "panglao" / "metadata" / "metadata_test.csv"


class LitDataPaths:
    cache = PRETRAIN_ROOT / "streaming" / "litdata" / "cache"
    data_cache = PRETRAIN_ROOT / "streaming" / "litdata" / "datacache"


class StreamingPanglaoPaths:
    rdata = PRETRAIN_ROOT / "panglao" / "rdata"
    h5 = PRETRAIN_ROOT / "streaming" / "panglao" / "h5"
    litdata = PRETRAIN_ROOT / "streaming" / "panglao" / "litdata"


class SNPdbPaths:
    raw = PRETRAIN_ROOT / "snpdb" / "raw"
    parquet_dir = PRETRAIN_ROOT / "snpdb" / "parquet"
    litdata_dir = PRETRAIN_ROOT / "snpdb" / "litdata"

    test_tabix = PRETRAIN_ROOT / "snpdb" / "GCF_000001405.25_test.gz"
    test_tabix_index = PRETRAIN_ROOT / "snpdb" / "GCF_000001405.25_test.gz.tbi"
    test_fasta = PRETRAIN_ROOT / "snpdb" / "hs37d5_test.fa"


class HiCPaths:
    raw = PRETRAIN_ROOT / "hic" / "raw"
    parquet_dir = PRETRAIN_ROOT / "hic" / "parquet"
    litdata_dir = PRETRAIN_ROOT / "hic" / "litdata"


class InsulationPaths:
    raw = PRETRAIN_ROOT / "insulation" / "raw"
    parquet_dir = PRETRAIN_ROOT / "insulation" / "parquet"
    litdata_dir = PRETRAIN_ROOT / "insulation" / "litdata"


class DNASeqCorePromoterPaths:
    processed_data_source = FINETUNE_ROOT / "core_promoter_prediction"
    label_dict_path = processed_data_source / "core_promoter_prediction_all_labels.json"


class DNASeqPromoterPaths:
    processed_data_source = FINETUNE_ROOT / "promoter_prediction" / "len_300" / "fold1"
    label_dict_path = processed_data_source / "promoter_prediction_all_labels.json"


class DNASeqMPRAPaths:
    processed_data_source = FINETUNE_ROOT / "lenti_mpra_regression" / "K562_original"
    label_dict_path = processed_data_source / "K562_original_all_labels.json"


class DNASeqSpliceSitePaths:
    processed_data_source = FINETUNE_ROOT / "splice_site_prediction"
    label_dict_path = processed_data_source / "splice_site_prediction_all_labels.json"


class DNASeqCovidPaths:
    processed_data_source = FINETUNE_ROOT / "covid_prediction"
    label_dict_path = processed_data_source / "covid_prediction_all_labels.json"


class DNASeqChromatinProfilePaths:
    processed_data_source = FINETUNE_ROOT / "chromatin_profile_prediction" / "len_1000"
    label_dict_path = (
        processed_data_source / "chromatin_profile_prediction_all_labels.json"
    )


class DNASeqSnp2TraitPaths:
    processed_data_source = FINETUNE_ROOT / "snp2trait"
    label_dict_path = processed_data_source / "snp2trait_all_labels.json"


class StreamingDNASeqSnp2TraitPaths:
    processed_data_source = FINETUNE_ROOT / "snp2trait" / "litdata"
    label_dict_path = (
        FINETUNE_ROOT / "snp2trait" / "snp2trait_prediction_all_labels.json"
    )


class DNASeqDrosophilaEnhancerPaths:
    processed_data_source = FINETUNE_ROOT / "drosophila_enhancer_prediction"
    label_dict_path = (
        processed_data_source / "drosophila_enhancer_prediction_all_labels.json"
    )


class DNASeqEpigeneticMarksPaths:
    processed_data_source = FINETUNE_ROOT / "epigenetic_marks_prediction" / "H4"

    label_dict_path = (
        processed_data_source / "epigenetic_marks_prediction_all_labels.json"
    )


class DNASeqTranscriptionFactorPaths:
    processed_data_source = FINETUNE_ROOT / "tf_prediction" / "human" / "fold1"
    label_dict_path = processed_data_source / "tf_prediction_all_labels.json"


class StreamingDNASeqChromatinProfilePaths:
    processed_data_source = (
        FINETUNE_ROOT
        / "chromatin_profile_prediction/len_1000/litdata_multilabels_mcs_ref/"
    )
    label_dict_path = "chromatin_profile_all_labels.json"


class CellXGenePaths:
    root = PRETRAIN_ROOT / "cellxgene"
    processed = root / "processed.h5ad"
    label_dict_path = PRETRAIN_ROOT / "cellxgene" / "cellxgene_labels.json"
    nexus_index_path = PRETRAIN_ROOT / "cellxgene" / "range_index"
    soma_path = PRETRAIN_ROOT / "cellxgene" / "cxg_subset" / "soma"
    soma_index_path = PRETRAIN_ROOT / "cellxgene" / "cxg_subset" / "index"


class ScperturbPerturbationPaths:
    root = FINETUNE_ROOT / "adamson_weissman"
    processed = root / "processed.h5ad"


class PerturbxPaths:
    root = FINETUNE_ROOT / "perturbx"
    replogle_shuffled = root / "shuffled_mini_replogle.h5ad"
    h1_shuffled = root / "shuffled_mini_h1.h5ad"
    replogle = root / "mini_replogle.h5ad"
    h1 = root / "mini_h1.h5ad"
    replogle_index_train = root / "shuffled_mini_replogle_train"
    replogle_index_dev = root / "shuffled_mini_replogle_dev"
    replogle_index_test = root / "shuffled_mini_replogle_test"
    h1_index_train = root / "shuffled_mini_h1_train"


class GearsPerturbationPaths:
    root = FINETUNE_ROOT / "norman"
    processed = root / "norman.h5ad"


class TcgaKircPaths:
    root = FINETUNE_ROOT / "TCGA_KIRC"
    gebc_processed = root / "gecb_survival_5y_tumor_only.h5ad"
    gesp_processed = root / "gesp_survival_tumor_only.h5ad"


test_pre_transforms: list[dict] = [
    {
        "transform_name": "RenameGenesTransform",
        "transform_args": {
            "gene_map": None,
        },
    },
    {
        "transform_name": "KeepGenesTransform",
        "transform_args": {"genes_to_keep": None},
    },
    {
        "transform_name": "NormalizeTotalTransform",
        "transform_args": {
            "exclude_highly_expressed": False,
            "max_fraction": 0.05,
            "target_sum": 10000.0,
        },
    },
    {
        "transform_name": "LogTransform",
        "transform_args": {"base": 2, "chunk_size": None, "chunked": None},
    },
    {
        "transform_name": "BinTransform",
        "transform_args": {"num_bins": 10, "binning_method": "int_cast"},
    },
]

test_perturbation_transforms: list[dict] = [
    {
        "transform_name": "BinTransform",
        "transform_args": {"num_bins": 50, "binning_method": "int_cast"},
    }
]


def _clear_directory_if_exists(dirname: Path):
    """Check if a directory exists and if it does, remove it with rm -r."""
    if dirname.exists():
        call(["rm", "-r", str(dirname)])


def clean_up_anndata_dataset_data():
    _clear_directory_if_exists(AnnCollectionDatasetPaths.root)
    _clear_directory_if_exists(ChangAllPaths.processed_ac)
    _clear_directory_if_exists(scIBD300kPaths.processed_ac)


def clean_up_litdata_cache_data():
    _clear_directory_if_exists(LitDataPaths.cache)
    _clear_directory_if_exists(LitDataPaths.data_cache)


def clean_up_streaming_snp_data():
    call(["rm", "-rf", str(Path(SNPdbPaths.parquet_dir))])
    call(["rm", "-rf", str(Path(SNPdbPaths.litdata_dir))])


def clean_up_streaming_hic_data():
    call(["rm", "-rf", str(Path(HiCPaths.parquet_dir))])
    call(["rm", "-rf", str(Path(HiCPaths.litdata_dir))])


def clean_up_streaming_insulation_data():
    call(["rm", "-rf", str(Path(InsulationPaths.parquet_dir))])
    call(["rm", "-rf", str(Path(InsulationPaths.litdata_dir))])


def clean_up_panglao_processed_data():
    _clear_directory_if_exists(PanglaoPaths.root / "h5ad")
    _clear_directory_if_exists(PanglaoPaths.root / "processed")


def clean_up_streaming_panglao_data():
    _clear_directory_if_exists(StreamingPanglaoPaths.h5)
    _clear_directory_if_exists(StreamingPanglaoPaths.litdata)


def clean_up_zheng68k_processed_data():
    _clear_directory_if_exists(Zheng68kPaths.processed)
    _clear_directory_if_exists(Zheng68kPaths.processed_raw_counts)
    _clear_directory_if_exists(Zheng68kPaths.label_dict_path)


def clean_up_scibd_processed_data():
    _clear_directory_if_exists(scIBDPaths.processed)
    _clear_directory_if_exists(scIBDPaths.label_dict_path)


def clean_up_scibd300k_processed_data():
    _clear_directory_if_exists(scIBD300kPaths.processed)
    _clear_directory_if_exists(scIBD300kPaths.label_dict_path)


def clean_up_humancellatlas_processed_data():
    _clear_directory_if_exists(HumanCellAtlasPaths.processed)
    _clear_directory_if_exists(HumanCellAtlasPaths.label_dict_path)


def clean_up_multiple_sclerosis_processed_data():
    _clear_directory_if_exists(MultipleSclerosisPaths.processed)
    _clear_directory_if_exists(MultipleSclerosisPaths.label_dict_path)


def clean_up_myeloid_processed_data():
    _clear_directory_if_exists(MyeloidPaths.processed)
    _clear_directory_if_exists(MyeloidPaths.label_dict_path)


def clean_up_cellxgene_processed_data():
    _clear_directory_if_exists(CellXGenePaths.processed)
    _clear_directory_if_exists(CellXGenePaths.label_dict_path)
    _clear_directory_if_exists(CellXGenePaths.nexus_index_path)


def clean_up_bulkrnaIBD_processed_data():
    _clear_directory_if_exists(BulkrnaIBDPaths.processed)
    _clear_directory_if_exists(BulkrnaIBDPaths.label_dict_path)


def clean_up_TIL_processed_data():
    _clear_directory_if_exists(TILPaths.processed)
    _clear_directory_if_exists(TILPaths.label_dict_path)


def clean_up_SCP1884_processed_data():
    _clear_directory_if_exists(SCP1884Paths.processed)
    _clear_directory_if_exists(SCP1884Paths.label_dict_path)


def clean_up_dnaseq_processed_data():
    # _clear_directory_if_exists(DNASeqPromoterPaths.processed_path)
    _clear_directory_if_exists(DNASeqPromoterPaths.label_dict_path)
    pass


def clean_up_pertrubx():
    call(["rm", "-rf", str(PerturbxPaths.replogle_shuffled)])
    call(["rm", "-rf", str(PerturbxPaths.h1_shuffled)])
    _clear_directory_if_exists(PerturbxPaths.h1_index_train)
    _clear_directory_if_exists(PerturbxPaths.replogle_index_train)
    _clear_directory_if_exists(PerturbxPaths.replogle_index_dev)
    _clear_directory_if_exists(PerturbxPaths.replogle_index_test)


def round_up_to_nearest_multiple_of_10(number):
    return ((number + 9) // 10) * 10


def get_gene_expression_tokenizations_for_data_module(dm):
    gene_id_count, expression_id_count = defaultdict(int), defaultdict(int)
    for batch in dm.train_dataloader():
        x = batch
        genes = x["input_ids"][:, 0, :].numpy()
        expressions = x["input_ids"][:, 1, :].numpy()
        for v, c in zip(*np.unique(genes, return_counts=True)):
            gene_id_count[v] += c
        for v, c in zip(*np.unique(expressions, return_counts=True)):
            expression_id_count[v] += c
    gene_token_count = {
        dm.tokenizer.convert_field_ids_to_tokens(field="genes", ids=int(k)): v
        for k, v in gene_id_count.items()
    }
    expression_token_count = {
        dm.tokenizer.convert_field_ids_to_tokens(field="expressions", ids=int(k)): v
        for k, v in expression_id_count.items()
    }
    gene_token_count.pop(dm.tokenizer.pad_token, None)
    expression_token_count.pop(dm.tokenizer.pad_token, None)
    return pd.Series(gene_token_count), pd.Series(expression_token_count)


def check_unk_levels_for_dm(dm, unk_frac_threshold=0.05):
    ds = dm.get_dataset_instance()
    ds_genes = ds.processed_data.var_names.to_numpy()
    tokenizer_genes = {*dm.tokenizer.get_field_vocab("genes")}
    mask = np.array([gene in tokenizer_genes for gene in ds_genes])
    captured_data = sum(ds.processed_data[:, mask].X.data)
    total_data = sum(ds.processed_data.X.data)
    unk_frac = 1 - captured_data / total_data
    assert (
        unk_frac < unk_frac_threshold
    ), f"Intolerable level of unknown gene tokens: {unk_frac}"


def check_h5ad_file_is_csr(adata):
    assert isinstance(adata.X, csr_matrix)
    return


def get_gene_label_tokenizations_for_t5_data_module(dm):
    gene_id_count, label_id_count = defaultdict(int), defaultdict(int)
    for batch in dm.train_dataloader():
        x = batch
        genes = x["data.encoder_input_token_ids"].numpy()
        labels = x["data.labels_token_ids"].numpy()
        for v, c in zip(*np.unique(genes, return_counts=True)):
            gene_id_count[v] += c
        for v, c in zip(*np.unique(labels, return_counts=True)):
            label_id_count[v] += c
    return pd.Series(gene_id_count), pd.Series(label_id_count)


def check_unk_levels_for_t5_dm(dm, unk_frac_threshold=0.05):
    gene_df, label_df = get_gene_label_tokenizations_for_t5_data_module(dm)

    unk_token_id = dm.tokenizer_op.get_token_id("<UNK>")
    unk_frac = gene_df.get(unk_token_id, 0) / sum(gene_df)

    assert (
        unk_frac < unk_frac_threshold
    ), f"Intolerable level of unknown gene tokens: {unk_frac}"
    assert unk_token_id not in label_df.index


def check_splits(
    h5ad_path,
    split_weights,
    test_needle,
    balancing_label,
    tol=0.1,
):
    metadata = read_h5ad(h5ad_path).obs
    splits = get_split_column(metadata, split_weights, balancing_label)
    assert {*splits.unique()} == {"train", "dev", "test"}
    assert splits.loc[test_needle] == "test"
    assert (
        split_weights["train"] - tol
        < (splits == "train").mean()
        < split_weights["train"] + tol
    )
    assert (
        split_weights["dev"] - tol
        < (splits == "dev").mean()
        < split_weights["dev"] + tol
    )
    assert (
        split_weights["test"] - tol
        < (splits == "dev").mean()
        < split_weights["test"] + tol
    )


def check_h5ad_file_structure(h5ad_path):
    ds = ad.read_h5ad(h5ad_path)
    assert ds.obs.index.name == "index", "the obs object Index should be named 'index'"

    def has_letter_and_number(idx):
        has_letter = any(char.isalpha() for char in idx)
        has_number = any(char.isnumeric() for char in idx)
        return has_letter and has_number

    at_least_one_mixed = any(has_letter_and_number(idx) for idx in ds.var_names)
    assert (
        at_least_one_mixed
    ), "There should be at least one gene name with both a letter and a number in its name"


def get_test_task_config(tmpdir, resume_training_from_ckpt=False):
    filename = "epoch={epoch}-step={step}-val_loss={validation/loss:.2f}"

    task_config = config.TrainingTaskConfig(
        default_root_dir=tmpdir,
        max_epochs=1,
        max_steps=3,
        accelerator="cpu",
        val_check_interval=3,
        gradient_clip_val=0.5,
        precision="32",
        enable_model_summary=False,
        enable_progress_bar=False,
        enable_checkpointing=True,
        resume_training_from_ckpt=resume_training_from_ckpt,
        detect_anomaly=False,
        callbacks=[
            ModelCheckpoint(
                dirpath=Path(tmpdir),
                save_last=True,
                save_top_k=0,
                filename=filename,
                auto_insert_metric_name=False,
            )
        ],
    )

    return task_config


def make_model_config_with_ckpt(trained_model_config, ckpt_path: str):
    model_config_dict = trained_model_config.to_dict()
    model_config_dict["fields"] = trained_model_config.fields
    model_config_dict["label_columns"] = trained_model_config.label_columns
    model_config_dict["checkpoint"] = ckpt_path
    model_config = trained_model_config.__class__(**model_config_dict)
    return model_config


def transform_and_return_args(
    paths, factory: base_rna_dataset.BaseRNAExpressionDataset, **kwargs
):
    stratifying_label = None
    if "stratifying_label" in kwargs:
        stratifying_label = kwargs.pop("stratifying_label")
    ds_kwargs = {
        "label_dict_path": paths.label_dict_path,
        **kwargs,
    }
    if (processed_attr := kwargs.get("processed_attr", None)) is not None:
        processed_path = getattr(paths, processed_attr)
    else:
        processed_path = paths.processed
    dt = DatasetTransformer(
        source_h5ad_file_name=paths.root / "h5ad" / factory.source_h5ad_file_name,
        split_weights=ds_kwargs.pop("split_weights", None),
        transforms=ds_kwargs.pop("transforms", None),
        split_column_name=ds_kwargs.get("split_column_name", None),
        stratifying_label=stratifying_label,
    )
    processed_data = dt.process_datasets()
    processed_data.write_h5ad(processed_path)
    ds_kwargs["processed_data_source"] = processed_path
    return ds_kwargs


def clean_processed_data():
    helpers.clean_up_panglao_processed_data()
    helpers.clean_up_streaming_panglao_data()
    helpers.clean_up_streaming_snp_data()
    helpers.clean_up_streaming_hic_data()
    helpers.clean_up_streaming_insulation_data()
    helpers.clean_up_zheng68k_processed_data()
    helpers.clean_up_scibd_processed_data()
    helpers.clean_up_scibd300k_processed_data()
    helpers.clean_up_humancellatlas_processed_data()
    helpers.clean_up_cellxgene_processed_data()
    helpers.clean_up_bulkrnaIBD_processed_data()
    helpers.clean_up_TIL_processed_data()
    helpers.clean_up_SCP1884_processed_data()
    helpers.clean_up_dnaseq_processed_data()
    helpers.clean_up_anndata_dataset_data()
    helpers.clean_up_pertrubx()


def ensure_expected_indexes_and_labels_present(
    dataset_kwargs, label_column_names, dataset_factory, sample_index=2
):
    test_ds = dataset_factory(**dataset_kwargs, split="test")
    train_ds = dataset_factory(**dataset_kwargs, split="train")
    dev_ds = dataset_factory(**dataset_kwargs, split="dev")

    test_mfi = test_ds[sample_index]
    train_mfi = train_ds[sample_index]
    dev_mfi = dev_ds[sample_index]
    expected_keys = {"cell_name", *label_column_names}

    assert test_mfi.metadata.keys() == expected_keys
    assert train_mfi.metadata.keys() == expected_keys
    assert dev_mfi.metadata.keys() == expected_keys

    assert test_mfi.metadata["cell_name"] in test_ds.processed_data.obs_names
    assert train_mfi.metadata["cell_name"] in train_ds.processed_data.obs_names
    assert dev_mfi.metadata["cell_name"] in dev_ds.processed_data.obs_names

    for lc in label_column_names:
        assert test_mfi.metadata[lc] in {*test_ds.metadata[lc]}
        assert train_mfi.metadata[lc] in {*train_ds.metadata[lc]}
        assert dev_mfi.metadata[lc] in {*dev_ds.metadata[lc]}


def update_label_columns(label_columns: list[LabelColumnInfo], label_dict):
    for label_column in label_columns:
        label_column.update_n_unique_values(label_dict)


def default_mlm_losses_from_fields(fields: list[config.FieldInfo]):
    """Create default MLM loss configurations for masked fields."""
    return [
        LossTask(
            source=FieldSource(field.field_name),
            objective=CrossEntropyObjective(),
            metrics=[{"name": "accuracy"}, {"name": "perplexity"}],
        )
        for field in fields
        if field.is_masked
    ]


def build_expected_metric_keys(
    losses: list[LossTask],
    splits: list[str] | None = None,
    include_perplexity: bool = True,
) -> set[str]:
    """Build expected metric keys from LossTask configurations."""
    splits = splits or ["train", "validation"]
    keys = set()

    # Group by metric_key
    metric_key_groups = {}
    for loss in losses:
        metric_key_groups.setdefault(loss.metric_key, []).append(loss)

    for split in splits:
        is_train = split == "train"
        suffixes = ["_step", "_epoch"] if is_train else [""]

        # Total loss
        for suffix in suffixes:
            keys.add(f"{split}/loss{suffix}")

        # Per-loss loss metrics
        for loss in losses:
            for suffix in suffixes:
                keys.add(f"{split}/{loss.name}_loss{suffix}")

        # Metrics grouped by metric_key
        for metric_key, group in metric_key_groups.items():
            # Collect unique metric names across group
            seen = set()
            for loss in group:
                for metric in loss._metrics or loss.objective.default_metrics():
                    if metric["name"] not in seen:
                        seen.add(metric["name"])
                        for suffix in suffixes:
                            keys.add(f"{split}/{metric_key}_{metric['name']}{suffix}")

    return keys


def assert_metrics_present(
    logged_metrics: dict, expected_metrics: list[str], allow_extra: bool = True
):
    """Assert that expected metrics are present in logged metrics."""
    logged, expected = set(logged_metrics.keys()), set(expected_metrics)

    if missing := expected - logged:
        raise AssertionError(f"Missing: {sorted(missing)}\nAvailable: {sorted(logged)}")

    if not allow_extra and (extra := logged - expected):
        raise AssertionError(f"Unexpected: {sorted(extra)}")
