import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pytorch_lightning.callbacks import ModelCheckpoint

from bmfm_targets import config
from bmfm_targets.config.main_config import (
    default_fields,
)
from bmfm_targets.datasets import sciplex3
from bmfm_targets.datasets.cellxgene.cellxgene_soma_dataset import (
    CellXGeneSOMADataModule,
)
from bmfm_targets.datasets.data_conversion.h5ad2litdata import convert_h5ad_to_litdata
from bmfm_targets.datasets.data_conversion.parquet2litdata import (
    convert_parquet_to_litdata,
)
from bmfm_targets.datasets.dnaseq import (
    DNASeqChromatinProfileDataModule,
    DNASeqCovidDataModule,
    DNASeqDrosophilaEnhancerDataModule,
    DNASeqEpigeneticMarksDataModule,
    DNASeqMPRADataModule,
    DNASeqPromoterDataModule,
    DNASeqSnp2TraitDataModule,
    DNASeqSpliceSiteDataModule,
    DNASeqTranscriptionFactorDataModule,
)
from bmfm_targets.datasets.panglaodb import PanglaoDBDataModule
from bmfm_targets.datasets.panglaodb.panglaodb_converter import (
    convert_all_rdatas_to_h5ad,
    create_sra_splits,
)
from bmfm_targets.datasets.perturbation import ScperturbDataModule
from bmfm_targets.datasets.SNPdb import (
    hic_splitter,
    insulation_splitter,
    snp_data_splitter,
)
from bmfm_targets.datasets.zheng68k import Zheng68kDataModule

# test_ function args are pytest fixtures defined in conftest.py`
from bmfm_targets.tasks.task_utils import make_trainer_for_task, train
from bmfm_targets.tests import helpers
from bmfm_targets.tests.helpers import (
    Zheng68kPaths,
    clean_processed_data,
    load_test_tokenizer,
)
from bmfm_targets.tokenization import (
    get_all_genes_tokenizer,
    get_gene2vec_tokenizer,
    get_snp2vec_BPEtokenizer,
    get_snp2vec_tokenizer,
    load_tokenizer,
)
from bmfm_targets.training import losses
from bmfm_targets.training.modules import (
    MLMTrainingModule,
    MultiTaskTrainingModule,
    SequenceClassificationTrainingModule,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _suppress_matplotlib_font_debug_logs():
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def fields():
    field_dicts = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "expressions",
            "pretrained_embedding": None,
            "is_masked": True,
            "vocab_update_strategy": "static",
            "decode_modes": ["token_scores"],
        },
    ]

    fields = [config.FieldInfo(**fd) for fd in field_dicts]
    return fields


@pytest.fixture(scope="session")
def dynamic_fields():
    field_dicts = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "dynamic",
        },
        {
            "field_name": "expressions",
            "pretrained_embedding": None,
            "is_masked": True,
            "vocab_update_strategy": "dynamic",
            "decode_modes": ["token_scores"],
        },
    ]

    fields = [config.FieldInfo(**fd) for fd in field_dicts]
    return fields


@pytest.fixture(scope="session")
def test_tokenizer_fields():
    test_field_dicts = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "expressions",
            "pretrained_embedding": None,
            "is_masked": True,
            "vocab_update_strategy": "static",
            "decode_modes": ["token_scores"],
        },
    ]
    test_fields = [config.FieldInfo(**fd) for fd in test_field_dicts]
    tokenizer = load_test_tokenizer()
    for field in test_fields:
        field.update_vocab_size(tokenizer)
    return test_fields


@pytest.fixture(scope="session")
def test_tokenizer_fields_multimask():
    test_field_dicts = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": True,
            "vocab_update_strategy": "static",
            "decode_modes": ["token_scores"],
        },
        {
            "field_name": "expressions",
            "pretrained_embedding": None,
            "is_masked": True,
            "vocab_update_strategy": "static",
            "decode_modes": ["token_scores"],
        },
    ]
    test_fields = [config.FieldInfo(**fd) for fd in test_field_dicts]
    tokenizer = load_test_tokenizer()
    for field in test_fields:
        field.update_vocab_size(tokenizer)
    return test_fields


@pytest.fixture(scope="session")
def gene2vec_fields():
    gene2vec_field_dicts = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "expressions",
            "pretrained_embedding": None,
            "is_masked": True,
            "vocab_update_strategy": "static",
            "decode_modes": ["token_scores"],
        },
    ]

    gene2vec_fields = [config.FieldInfo(**fd) for fd in gene2vec_field_dicts]
    tokenizer = get_gene2vec_tokenizer()
    for field in gene2vec_fields:
        field.update_vocab_size(tokenizer)
    return gene2vec_fields


@pytest.fixture(scope="session")
def gene2vec_unmasked_fields():
    gene2vec_field_dicts = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "expressions",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
    ]

    gene2vec_unmasked_fields = [config.FieldInfo(**fd) for fd in gene2vec_field_dicts]
    tokenizer = get_gene2vec_tokenizer()
    for field in gene2vec_unmasked_fields:
        field.update_vocab_size(tokenizer)
    return gene2vec_unmasked_fields


@pytest.fixture(scope="session")
def all_genes_fields():
    field_dicts = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "expressions",
            "pretrained_embedding": None,
            "is_masked": True,
            "vocab_update_strategy": "static",
            "decode_modes": ["token_scores"],
        },
    ]

    fields = [config.FieldInfo(**fd) for fd in field_dicts]
    tokenizer = get_all_genes_tokenizer()
    for field in fields:
        field.update_vocab_size(tokenizer)
    return fields


@pytest.fixture(scope="session")
def perturbation_fields_cve():
    perturbation_field_dicts = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "expressions",
            "is_masked": False,
            "decode_modes": ["regression", "is_zero"],
            "tokenization_strategy": "continuous_value_encoder",
            "vocab_update_strategy": "static",
            "encoder_kwargs": {
                "kind": "mlp_with_special_token_embedding",
                "zero_as_special_token": True,
            },
        },
        {
            "field_name": "perturbations",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "label_expressions",
            "is_masked": False,
            "is_input": False,
            "decode_modes": ["regression", "is_zero"],
            "tokenization_strategy": "continuous_value_encoder",
            "vocab_update_strategy": "static",
            "encoder_kwargs": {
                "kind": "mlp_with_special_token_embedding",
                "zero_as_special_token": True,
            },
        },
    ]

    perturbation_fields = [config.FieldInfo(**fd) for fd in perturbation_field_dicts]
    tokenizer = get_all_genes_tokenizer()
    for field in perturbation_fields:
        # TODO: this change is needed if we may have a perubation_field that should not be tokenized
        # but rather passed as is.  If not, then we lose the exception if we used the wrong field name
        # Not sure what the correct option should be
        if field.field_name in tokenizer.tokenizers.keys():
            field.update_vocab_size(tokenizer)
    return perturbation_fields


@pytest.fixture(scope="session")
def perturbation_fields_tokenized():
    perturbation_field_dicts = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "expressions",
            "is_masked": False,
            "decode_modes": ["regression", "token_scores"],
        },
        {
            "field_name": "perturbations",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "label_expressions",
            "is_masked": False,
            "is_input": False,
            "decode_modes": ["regression", "token_scores"],
        },
    ]

    perturbation_fields = [config.FieldInfo(**fd) for fd in perturbation_field_dicts]
    tokenizer = get_gene2vec_tokenizer()
    for field in perturbation_fields:
        # TODO: this change is needed if we may have a perubation_field that should not be tokenized
        # but rather passed as is.  If not, then we lose the exception if we used the wrong field name
        # Not sure what the correct option should be
        if field.field_name in tokenizer.tokenizers.keys():
            field.update_vocab_size(tokenizer)
    return perturbation_fields


@pytest.fixture(scope="session")
def label_expression_fields_gene2vec():
    perturbation_field_dicts = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "expressions",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "label_expressions",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
            "is_input": False,
            "decode_modes": ["regression", "is_zero"],
        },
    ]

    perturbation_fields = [config.FieldInfo(**fd) for fd in perturbation_field_dicts]
    tokenizer = load_tokenizer("gene2vec")
    for field in perturbation_fields:
        if field.is_input:
            field.update_vocab_size(tokenizer)
    return perturbation_fields


@pytest.fixture(scope="session")
def all_genes_fields_with_regression_masking():
    all_genes_fields_with_regression_masking = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": True,
            "vocab_update_strategy": "static",
            "decode_modes": ["token_scores"],
        },
        {
            "field_name": "expressions",
            "pretrained_embedding": None,
            "is_masked": True,
            "decode_modes": ["regression", "is_zero"],
            "tokenization_strategy": "continuous_value_encoder",
            "vocab_update_strategy": "static",
            "encoder_kwargs": {
                "kind": "mlp_with_special_token_embedding",
                "zero_as_special_token": True,
            },
        },
    ]

    all_genes_fields_with_regression_masking_dicts = [
        config.FieldInfo(**fd) for fd in all_genes_fields_with_regression_masking
    ]
    tokenizer = load_tokenizer("all_genes")
    for field in all_genes_fields_with_regression_masking_dicts:
        if field.is_input:
            field.update_vocab_size(tokenizer)
    return all_genes_fields_with_regression_masking_dicts


@pytest.fixture(scope="session")
def all_genes_fields_with_regression_masking_plus_mvc():
    all_genes_fields_with_regression_masking = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "expressions",
            "pretrained_embedding": None,
            "is_masked": True,
            "decode_modes": ["regression", "is_zero", "mvc_regression", "mvc_is_zero"],
            "tokenization_strategy": "continuous_value_encoder",
            "vocab_update_strategy": "static",
            "encoder_kwargs": {
                "kind": "mlp_with_special_token_embedding",
                "zero_as_special_token": True,
            },
        },
    ]

    all_genes_fields_with_regression_masking_dicts = [
        config.FieldInfo(**fd) for fd in all_genes_fields_with_regression_masking
    ]
    tokenizer = load_tokenizer("all_genes")
    for field in all_genes_fields_with_regression_masking_dicts:
        if field.is_input:
            field.update_vocab_size(tokenizer)
    return all_genes_fields_with_regression_masking_dicts


@pytest.fixture(scope="session")
def all_genes_fields_with_rda_regression_masking():
    all_genes_fields_with_rda_regression_masking = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "expressions",
            "pretrained_embedding": None,
            "is_masked": True,
            "decode_modes": ["regression", "is_zero"],
            "tokenization_strategy": "continuous_value_encoder",
            "vocab_update_strategy": "static",
            "encoder_kwargs": {
                "kind": "mlp_with_special_token_embedding",
                "zero_as_special_token": True,
            },
        },
        {
            "field_name": "label_expressions",
            "pretrained_embedding": None,
            "is_masked": False,
            "decode_modes": ["regression", "is_zero"],
            "tokenization_strategy": "continuous_value_encoder",
            "vocab_update_strategy": "static",
            "is_input": False,
        },
    ]

    all_genes_fields_with_rda_regression_masking_dicts = [
        config.FieldInfo(**fd) for fd in all_genes_fields_with_rda_regression_masking
    ]
    tokenizer = load_tokenizer("all_genes")
    for field in all_genes_fields_with_rda_regression_masking_dicts:
        if field.is_input:
            field.update_vocab_size(tokenizer)
    return all_genes_fields_with_rda_regression_masking_dicts


@pytest.fixture(scope="session")
def gene2vec_fields_regression_no_tokenization():
    gene2vec_field_dicts = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "decode_modes": ["token_scores"],
            "is_masked": True,
            "vocab_update_strategy": "static",
        },
        {
            "field_name": "expressions",
            "pretrained_embedding": None,
            "is_masked": True,
            "decode_modes": ["regression", "is_zero"],
            "vocab_update_strategy": "static",
            "tokenization_strategy": "continuous_value_encoder",
            "encoder_kwargs": {
                "kind": "mlp",  # _with_special_token_embedding",
                # "zero_as_special_token": True,
            },
        },
    ]

    gene2vec_fields = [config.FieldInfo(**fd) for fd in gene2vec_field_dicts]
    tokenizer = get_gene2vec_tokenizer()
    for field in gene2vec_fields:
        field.update_vocab_size(tokenizer)

    return gene2vec_fields


@pytest.fixture(scope="session")
def gene2vec_fields_regression_with_tokenization():
    gene2vec_field_dicts = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": True,
            "vocab_update_strategy": "static",
            "decode_modes": ["token_scores"],
        },
        {
            "field_name": "expressions",
            "pretrained_embedding": None,
            "is_masked": True,
            "decode_modes": ["regression", "token_scores"],
            "vocab_update_strategy": "static",
        },
    ]

    gene2vec_fields = [config.FieldInfo(**fd) for fd in gene2vec_field_dicts]
    tokenizer = get_gene2vec_tokenizer()
    for field in gene2vec_fields:
        field.update_vocab_size(tokenizer)
    return gene2vec_fields


@pytest.fixture(scope="session")
def geneformer_gene2vec_fields():
    gene2vec_field_dicts = [
        {
            "field_name": "genes",
            "pretrained_embedding": None,
            "is_masked": True,
            "vocab_update_strategy": "static",
            "decode_modes": ["token_scores"],
        },
    ]
    gene2vec_fields = [config.FieldInfo(**fd) for fd in gene2vec_field_dicts]
    tokenizer = get_gene2vec_tokenizer()
    for field in gene2vec_fields:
        field.update_vocab_size(tokenizer)
    return gene2vec_fields


@pytest.fixture(scope="session")
def zheng68k_label_columns():
    return [
        config.LabelColumnInfo(
            label_column_name="celltype", is_stratification_label=True
        )
    ]


@pytest.fixture(scope="session")
def sciplex3_label_columns():
    return [config.LabelColumnInfo(label_column_name="cell_type")]


@pytest.fixture(scope="session")
def _convert_rdata_to_litdata():
    helpers.initialize_litdata()
    data_info_path = str(helpers.PanglaoPaths.root / "metadata" / "metadata_test.csv")
    rdata_dir = str(helpers.PanglaoPaths.root / "rdata")
    dataframe = create_sra_splits(data_info_path, rdata_dir, filter_query=None)

    convert_all_rdatas_to_h5ad(
        helpers.StreamingPanglaoPaths.h5, dataframe, num_workers=0
    )

    convert_h5ad_to_litdata(
        helpers.StreamingPanglaoPaths.h5,
        helpers.StreamingPanglaoPaths.litdata,
    )


@pytest.fixture(scope="session")
def streaming_panglao_parameters(gene2vec_fields):
    dataset_kwargs = {"input_dir": helpers.StreamingPanglaoPaths.litdata}
    tokenizer = get_gene2vec_tokenizer()
    pars = {
        "tokenizer": tokenizer,
        "batch_size": 2,
        "fields": gene2vec_fields,
        "num_workers": 2,
        "mlm": True,
        "collation_strategy": "language_modeling",
        "dataset_kwargs": dataset_kwargs,
    }
    return pars


@pytest.fixture(scope="session")
def snp2vec_fields():
    snp2vec_field_dicts = [
        {
            "field_name": "dna_chunks",
            "pretrained_embedding": None,
            "is_masked": True,
            "vocab_update_strategy": "static",
            "decode_modes": ["token_scores"],
        },
    ]

    snp2vec_fields = [config.FieldInfo(**fd) for fd in snp2vec_field_dicts]
    tokenizer = get_snp2vec_tokenizer()
    for field in snp2vec_fields:
        field.update_vocab_size(tokenizer)
    return snp2vec_fields


@pytest.fixture(scope="session")
def hic_fields():
    hic_field_dicts = [
        {
            "field_name": "dna_chunks",
            "pretrained_embedding": None,
            "is_masked": True,
            "vocab_update_strategy": "static",
            "decode_modes": ["token_scores"],
        },
        {
            "field_name": "dna_chunk_ids",
            "pretrained_embedding": None,
            "is_masked": True,
            "vocab_update_strategy": "static",
            "decode_modes": ["token_scores"],
        },
    ]

    hic_fields = [config.FieldInfo(**fd) for fd in hic_field_dicts]
    tokenizer = get_snp2vec_tokenizer()
    for field in hic_fields:
        field.update_vocab_size(tokenizer)
    return hic_fields


@pytest.fixture(scope="session")
def finetuning_snp2vec_fields():
    snp2vec_field_dicts = [
        {
            "field_name": "dna_chunks",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
    ]

    snp2vec_fields = [config.FieldInfo(**fd) for fd in snp2vec_field_dicts]
    tokenizer = get_snp2vec_tokenizer()
    for field in snp2vec_fields:
        field.update_vocab_size(tokenizer)
    return snp2vec_fields


@pytest.fixture(scope="session")
def _convert_raw_to_lit():
    helpers.initialize_litdata()
    input_path = Path(helpers.SNPdbPaths.raw)
    snp_data_splitter.split(
        input_path=str(input_path),
        output_path=helpers.SNPdbPaths.parquet_dir,
    )
    convert_parquet_to_litdata(
        helpers.SNPdbPaths.parquet_dir,
        helpers.SNPdbPaths.litdata_dir,
        get_snp2vec_BPEtokenizer(),
        num_workers=0,
    )


@pytest.fixture(scope="session")
def streaming_snpdb_parameters(snp2vec_fields):
    dataset_kwargs = {"input_dir": helpers.SNPdbPaths.litdata_dir}
    tokenizer = get_snp2vec_tokenizer()
    pars = {
        "tokenizer": tokenizer,
        "batch_size": 2,
        "fields": snp2vec_fields,
        "num_workers": 0,
        "mlm": True,
        "collation_strategy": "language_modeling",
        "dataset_kwargs": dataset_kwargs,
    }
    return pars


@pytest.fixture(scope="session")
def combined_streaming_snpdb_parameters(snp2vec_fields):
    ## TODO simply duplicate the litdata for now
    ## TODO add test weights = [0.5, 0.5], iterate_over_all=False
    dataset_kwargs = {
        "input_dirs": [helpers.SNPdbPaths.litdata_dir, helpers.SNPdbPaths.litdata_dir]
    }
    tokenizer = get_snp2vec_tokenizer()
    pars = {
        "tokenizer": tokenizer,
        "batch_size": 2,
        "fields": snp2vec_fields,
        "num_workers": 2,
        "mlm": True,
        "collation_strategy": "language_modeling",
        "dataset_kwargs": dataset_kwargs,
    }
    return pars


@pytest.fixture(scope="session")
def _convert_hic_raw_to_lit():
    helpers.initialize_litdata()
    input_path = Path(helpers.HiCPaths.raw)
    hic_splitter.split(
        input_path=str(input_path),
        output_path=helpers.HiCPaths.parquet_dir,
    )
    convert_parquet_to_litdata(
        helpers.HiCPaths.parquet_dir,
        helpers.HiCPaths.litdata_dir,
        get_snp2vec_BPEtokenizer(),
        data_source="hic",
        num_workers=0,
    )


@pytest.fixture(scope="session")
def hic_label_columns():
    hic_label_dicts = [
        {
            "label_column_name": "hic_contact",
            "is_regression_label": False,
            "n_unique_values": 2,
        }
    ]
    hic_label_columns = [config.LabelColumnInfo(**ld) for ld in hic_label_dicts]
    return hic_label_columns


@pytest.fixture(scope="session")
def streaming_hic_parameters(hic_fields, hic_label_columns):
    dataset_kwargs = {
        "input_dir": helpers.HiCPaths.litdata_dir,
    }
    tokenizer = get_snp2vec_tokenizer()
    pars = {
        "tokenizer": tokenizer,
        "batch_size": 10,
        "fields": hic_fields,
        "label_columns": hic_label_columns,
        "max_length": 512,
        "num_workers": 0,
        "mlm": True,
        "collation_strategy": "multitask",
        "dataset_kwargs": dataset_kwargs,
    }
    return pars


@pytest.fixture(scope="session")
def streaming_hic_only_parameters(hic_fields, hic_label_columns):
    dataset_kwargs = {
        "input_dir": helpers.HiCPaths.litdata_dir,
    }
    tokenizer = get_snp2vec_tokenizer()
    pars = {
        "tokenizer": tokenizer,
        "batch_size": 10,
        "fields": hic_fields,
        "label_columns": hic_label_columns,
        "max_length": 512,
        "num_workers": 0,
        "mlm": False,
        "collation_strategy": "sequence_classification",
        "dataset_kwargs": dataset_kwargs,
    }
    return pars


@pytest.fixture(scope="session")
def _convert_insulation_raw_to_lit():
    helpers.initialize_litdata()
    input_path = Path(helpers.InsulationPaths.raw)
    insulation_splitter.split(
        input_path=str(input_path),
        output_path=helpers.InsulationPaths.parquet_dir,
    )
    convert_parquet_to_litdata(
        helpers.InsulationPaths.parquet_dir,
        helpers.InsulationPaths.litdata_dir,
        get_snp2vec_BPEtokenizer(),
        data_source="insulation",
        num_workers=0,
    )


@pytest.fixture(scope="session")
def insulation_fields():
    insulation_field_dicts = [
        {
            "field_name": "dna_chunks",
            "pretrained_embedding": None,
            "is_masked": True,
            "vocab_update_strategy": "static",
            "decode_modes": ["token_scores"],
        },
    ]

    insulation_fields = [config.FieldInfo(**fd) for fd in insulation_field_dicts]
    tokenizer = get_snp2vec_tokenizer()
    for field in insulation_fields:
        field.update_vocab_size(tokenizer)
    return insulation_fields


@pytest.fixture(scope="session")
def insulation_label_columns():
    insulation_label_dicts = [
        {
            "label_column_name": "insulation",
            "is_regression_label": True,
        }
    ]
    insulation_label_columns = [
        config.LabelColumnInfo(**ld) for ld in insulation_label_dicts
    ]
    return insulation_label_columns


@pytest.fixture(scope="session")
def streaming_insulation_parameters(insulation_fields, insulation_label_columns):
    dataset_kwargs = {
        "input_dir": helpers.InsulationPaths.litdata_dir,
    }
    tokenizer = get_snp2vec_tokenizer()
    pars = {
        "tokenizer": tokenizer,
        "batch_size": 10,
        "fields": insulation_fields,
        "label_columns": insulation_label_columns,
        "max_length": 512,
        "num_workers": 0,
        "mlm": True,
        "collation_strategy": "multitask",
        "dataset_kwargs": dataset_kwargs,
    }
    return pars


@pytest.fixture(scope="session")
def _panglao_convert_rdata_and_transform(gene2vec_fields):
    panglao_dataset_kwargs = {
        "data_dir": helpers.PanglaoPaths.root,
        "data_info_path": helpers.PanglaoPaths.test_metadata,
        "num_workers": 0,
        "transform_datasets": True,
        "convert_rdata_to_h5ad": True,
        "pre_transforms": helpers.test_pre_transforms,
    }

    kwargs: dict[str, Any] = {
        "num_workers": 0,
        "collation_strategy": "language_modeling",
        "mlm": True,
        "dataset_kwargs": panglao_dataset_kwargs,
    }

    tokenizer = get_gene2vec_tokenizer()
    pl_data_module_panglao = PanglaoDBDataModule(
        tokenizer=tokenizer,
        batch_size=3,
        fields=gene2vec_fields,
        max_length=512,
        limit_dataset_samples=12,
        **kwargs,
    )
    pl_data_module_panglao.prepare_data()
    pl_data_module_panglao.setup("fit")


@pytest.fixture(scope="session")
def pl_data_module_panglao_all_genes_tokenizer(
    all_genes_fields, _panglao_convert_rdata_and_transform
):
    panglao_dataset_kwargs = {
        "data_dir": helpers.PanglaoPaths.root,
        "data_info_path": helpers.PanglaoPaths.test_metadata,
        "filter_query": 'Species == "Homo sapiens"',
        "num_workers": 0,
        "transform_datasets": False,
        "convert_rdata_to_h5ad": False,
    }
    kwargs: dict[str, Any] = {
        "num_workers": 0,
        "collation_strategy": "language_modeling",
        "mlm": True,
        "dataset_kwargs": panglao_dataset_kwargs,
    }

    tokenizer = get_all_genes_tokenizer()
    pl_data_module_panglao = PanglaoDBDataModule(
        tokenizer=tokenizer,
        batch_size=3,
        fields=all_genes_fields,
        max_length=16,
        limit_dataset_samples=12,
        **kwargs,
    )
    pl_data_module_panglao.prepare_data()
    pl_data_module_panglao.setup("fit")
    return pl_data_module_panglao


@pytest.fixture(scope="session")
def pl_data_module_panglao(gene2vec_fields, _panglao_convert_rdata_and_transform):
    panglao_dataset_kwargs = {
        "data_dir": helpers.PanglaoPaths.root,
        "data_info_path": helpers.PanglaoPaths.test_metadata,
        "filter_query": 'Species == "Homo sapiens"',
        "num_workers": 0,
        "transform_datasets": False,
        "convert_rdata_to_h5ad": False,
    }

    kwargs: dict[str, Any] = {
        "num_workers": 0,
        "mlm": True,
        "collation_strategy": "language_modeling",
        "dataset_kwargs": panglao_dataset_kwargs,
    }

    tokenizer = get_gene2vec_tokenizer()
    pl_data_module_panglao = PanglaoDBDataModule(
        tokenizer=tokenizer,
        batch_size=3,
        fields=gene2vec_fields,
        max_length=16,
        limit_dataset_samples=12,
        **kwargs,
    )
    pl_data_module_panglao.prepare_data()
    pl_data_module_panglao.setup("fit")
    return pl_data_module_panglao


@pytest.fixture(scope="session")
def pl_data_module_panglao_protein_coding(
    all_genes_fields, _panglao_convert_rdata_and_transform
):
    panglao_dataset_kwargs = {
        "data_dir": helpers.PanglaoPaths.root,
        "data_info_path": helpers.PanglaoPaths.test_metadata,
        "filter_query": 'Species == "Homo sapiens"',
        "num_workers": 0,
        "transform_datasets": False,
        "convert_rdata_to_h5ad": False,
    }

    kwargs: dict[str, Any] = {
        "num_workers": 0,
        "mlm": True,
        "collation_strategy": "language_modeling",
        "dataset_kwargs": panglao_dataset_kwargs,
    }

    tokenizer = load_tokenizer("all_genes")
    pl_data_module_panglao = PanglaoDBDataModule(
        tokenizer=tokenizer,
        batch_size=3,
        fields=all_genes_fields,
        max_length=16,
        limit_dataset_samples=12,
        limit_genes="protein_coding",
        **kwargs,
    )
    pl_data_module_panglao.prepare_data()
    pl_data_module_panglao.setup("fit")
    return pl_data_module_panglao


@pytest.fixture(scope="session")
def pl_data_module_panglao_limit10(
    gene2vec_fields, _panglao_convert_rdata_and_transform
):
    dataset_kwargs = {
        "data_dir": helpers.PanglaoPaths.root,
        "data_info_path": helpers.PanglaoPaths.test_metadata,
        "filter_query": 'Species == "Homo sapiens"',
        "num_workers": 0,
        "transform_datasets": False,
        "convert_rdata_to_h5ad": False,
        "pre_transforms": helpers.test_pre_transforms,
    }

    kwargs: dict[str, Any] = {
        "num_workers": 0,
        "collation_strategy": "language_modeling",
        "mlm": True,
        "dataset_kwargs": dataset_kwargs,
    }

    tokenizer = get_gene2vec_tokenizer()
    dm = PanglaoDBDataModule(
        tokenizer=tokenizer,
        batch_size=3,
        fields=gene2vec_fields,
        limit_dataset_samples=10,
        **kwargs,
    )
    dm.prepare_data()
    dm.setup("fit")
    return dm


@pytest.fixture(scope="session")
def pl_data_module_panglao_regression(
    gene2vec_fields_regression_with_tokenization, _panglao_convert_rdata_and_transform
):
    panglao_dataset_kwargs = {
        "data_dir": helpers.PanglaoPaths.root,
        "data_info_path": helpers.PanglaoPaths.test_metadata,
        "filter_query": 'Species == "Homo sapiens"',
        "num_workers": 0,
        "transform_datasets": False,
        "convert_rdata_to_h5ad": False,
    }

    kwargs: dict[str, Any] = {
        "num_workers": 0,
        "mlm": True,
        "collation_strategy": "language_modeling",
        "dataset_kwargs": panglao_dataset_kwargs,
    }

    tokenizer = get_gene2vec_tokenizer()
    pl_data_module_panglao = PanglaoDBDataModule(
        tokenizer=tokenizer,
        batch_size=3,
        fields=gene2vec_fields_regression_with_tokenization,
        **kwargs,
    )
    pl_data_module_panglao.prepare_data()
    pl_data_module_panglao.setup("fit")
    yield pl_data_module_panglao

    helpers.clean_up_panglao_processed_data()


@pytest.fixture(scope="session")
def pl_data_module_panglao_rda(
    all_genes_fields_with_rda_regression_masking, _panglao_convert_rdata_and_transform
):
    panglao_dataset_kwargs = {
        "data_dir": helpers.PanglaoPaths.root,
        "data_info_path": helpers.PanglaoPaths.test_metadata,
        "filter_query": 'Species == "Homo sapiens"',
        "num_workers": 0,
        "expose_zeros": "all",
        "transform_datasets": True,
        "convert_rdata_to_h5ad": False,
        "processed_dir_name": "processed_rda",
        "pre_transforms": [
            {
                "transform_name": "RenameGenesTransform",
                "transform_args": {"gene_map": None},
            },
            {
                "transform_name": "KeepGenesTransform",
                "transform_args": {"genes_to_keep": None},
            },
            {
                "transform_name": "FilterCellsTransform",
                "transform_args": {"min_counts": 2},
            },
            {
                "transform_name": "FilterGenesTransform",
                "transform_args": {"min_counts": 1},
            },
        ],
    }

    kwargs: dict[str, Any] = {
        "num_workers": 0,
        "mlm": True,
        "collation_strategy": "language_modeling",
        "dataset_kwargs": panglao_dataset_kwargs,
        "rda_transform": "downsample",
        "switch_ratio": 0.0,
        "max_length": 16,
    }

    tokenizer = get_all_genes_tokenizer()
    pl_data_module_panglao = PanglaoDBDataModule(
        tokenizer=tokenizer,
        batch_size=4,
        fields=all_genes_fields_with_rda_regression_masking,
        **kwargs,
    )
    pl_data_module_panglao.prepare_data()
    pl_data_module_panglao.setup("fit")
    return pl_data_module_panglao


@pytest.fixture(scope="session")
def _zheng68k_transform_raw_counts(
    all_genes_fields_with_regression_masking, zheng68k_label_columns
):
    """Pre-transform fixture to avoid race conditions."""
    helpers.initialize_litdata()
    tokenizer = load_tokenizer("protein_coding")
    dm = Zheng68kDataModule(
        data_dir=helpers.Zheng68kPaths.root,
        processed_name=helpers.Zheng68kPaths.raw_counts_name,
        transform_kwargs={"transforms": []},
        transform_datasets=True,
        tokenizer=tokenizer,
        label_columns=zheng68k_label_columns,
        batch_size=2,
        fields=all_genes_fields_with_regression_masking,
        limit_dataset_samples=8,
        mlm=True,
        collation_strategy="language_modeling",
        rda_transform=2000,
        max_length=20,
        pad_to_multiple_of=2,
    )
    dm.prepare_data()


@pytest.fixture(scope="session")
def pl_zheng_mlm_raw_counts(
    all_genes_fields_with_regression_masking,
    zheng68k_label_columns,
    _zheng68k_transform_raw_counts,
):
    tokenizer = load_tokenizer("protein_coding")
    dm = Zheng68kDataModule(
        data_dir=helpers.Zheng68kPaths.root,
        processed_name=helpers.Zheng68kPaths.raw_counts_name,
        transform_kwargs={"transforms": []},
        transform_datasets=False,
        tokenizer=tokenizer,
        label_columns=zheng68k_label_columns,
        batch_size=2,
        fields=all_genes_fields_with_regression_masking,
        limit_dataset_samples=8,
        mlm=True,
        collation_strategy="language_modeling",
        rda_transform=2000,
        max_length=20,
        pad_to_multiple_of=2,
    )
    dm.prepare_data()
    dm.setup("fit")
    return dm


@pytest.fixture(scope="session")
def pl_data_module_panglao_regression_raw_counts(
    gene2vec_fields_regression_no_tokenization, _panglao_convert_rdata_and_transform
):
    panglao_dataset_kwargs = {
        "data_dir": helpers.PanglaoPaths.root,
        "data_info_path": helpers.PanglaoPaths.test_metadata,
        "filter_query": 'Species == "Homo sapiens"',
        "num_workers": 0,
        "transform_datasets": True,
        "convert_rdata_to_h5ad": False,
        "processed_dir_name": "processed_raw_counts",
        "pre_transforms": [
            {
                "transform_name": "RenameGenesTransform",
                "transform_args": {"gene_map": None},
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
                "transform_name": "FilterCellsTransform",
                "transform_args": {"min_counts": 2},
            },
            {
                "transform_name": "FilterGenesTransform",
                "transform_args": {"min_counts": 1},
            },
        ],
    }

    kwargs: dict[str, Any] = {
        "num_workers": 0,
        "mlm": True,
        "collation_strategy": "language_modeling",
        "dataset_kwargs": panglao_dataset_kwargs,
    }

    tokenizer = get_gene2vec_tokenizer()
    pl_data_module_panglao = PanglaoDBDataModule(
        tokenizer=tokenizer,
        batch_size=3,
        fields=gene2vec_fields_regression_no_tokenization,
        **kwargs,
    )
    pl_data_module_panglao.prepare_data()
    pl_data_module_panglao.setup("fit")
    return pl_data_module_panglao


@pytest.fixture(scope="session")
def pl_data_module_panglao_dynamic_binning(
    dynamic_fields, _panglao_convert_rdata_and_transform
):
    kwargs: dict[str, Any] = {
        "num_workers": 0,
        "collation_strategy": "language_modeling",
        "mlm": True,
        "dataset_kwargs": {
            "data_dir": helpers.PanglaoPaths.root,
            "data_info_path": helpers.PanglaoPaths.test_metadata,
            "filter_query": 'Species == "Homo sapiens"',
            "num_workers": 0,
            "transform_datasets": True,
            "convert_rdata_to_h5ad": False,
            "processed_dir_name": "processed_dynamic",
            "pre_transforms": [
                {
                    "transform_name": "RenameGenesTransform",
                    "transform_args": {"gene_map": None},
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
                    "transform_args": {"binning_method": "int_cast"},
                },
                {
                    "transform_name": "FilterCellsTransform",
                    "transform_args": {"min_counts": 2},
                },
                {
                    "transform_name": "FilterGenesTransform",
                    "transform_args": {"min_counts": 1},
                },
            ],
        },
    }

    tokenizer = get_gene2vec_tokenizer()
    dm = PanglaoDBDataModule(
        tokenizer=tokenizer,
        batch_size=3,
        fields=dynamic_fields,
        **kwargs,
    )
    dm.prepare_data()
    dm.setup("fit")
    return dm


@pytest.fixture(scope="session")
def pl_data_module_panglao_geneformer(
    geneformer_gene2vec_fields, _panglao_convert_rdata_and_transform
):
    kwargs: dict[str, Any] = {
        "num_workers": 0,
        "collation_strategy": "language_modeling",
        "mlm": True,
        "dataset_kwargs": {
            "data_dir": helpers.PanglaoPaths.root,
            "data_info_path": helpers.PanglaoPaths.test_metadata,
            "filter_query": 'Species == "Homo sapiens"',
            "num_workers": 0,
            "transform_datasets": True,
            "convert_rdata_to_h5ad": False,
            "processed_dir_name": "processed_ranked",
            "pre_transforms": [
                {
                    "transform_name": "RenameGenesTransform",
                    "transform_args": {"gene_map": None},
                },
                {
                    "transform_name": "KeepGenesTransform",
                    "transform_args": {"genes_to_keep": None},
                },
                {
                    "transform_name": "FilterCellsTransform",
                    "transform_args": {"min_counts": 2},
                },
                {
                    "transform_name": "FilterGenesTransform",
                    "transform_args": {"min_counts": 1},
                },
            ],
        },
    }

    tokenizer = get_gene2vec_tokenizer()
    pl_data_module_panglao_geneformer = PanglaoDBDataModule(
        tokenizer=tokenizer,
        batch_size=3,
        fields=geneformer_gene2vec_fields,
        sequence_order="sorted",
        **kwargs,
    )
    pl_data_module_panglao_geneformer.prepare_data()
    pl_data_module_panglao_geneformer.setup("fit")
    return pl_data_module_panglao_geneformer


@pytest.fixture(scope="session")
def _zheng68k_transform_default(gene2vec_unmasked_fields, zheng68k_label_columns):
    """Pre-transform fixture for default processed directory to avoid race conditions."""
    tokenizer = get_gene2vec_tokenizer()
    pl_data_module = Zheng68kDataModule(
        data_dir=helpers.Zheng68kPaths.root,
        tokenizer=tokenizer,
        transform_datasets=True,
        collation_strategy="sequence_classification",
        num_workers=0,
        batch_size=3,
        fields=gene2vec_unmasked_fields,
        label_columns=zheng68k_label_columns,
        max_length=16,
        limit_dataset_samples={"train": 12, "dev": 12, "predict": 2},
    )
    pl_data_module.prepare_data()


@pytest.fixture(scope="session")
def pl_data_module_zheng68k_seq_cls(
    gene2vec_unmasked_fields, zheng68k_label_columns, _zheng68k_transform_default
):
    tokenizer = get_gene2vec_tokenizer()
    pl_data_module = Zheng68kDataModule(
        data_dir=helpers.Zheng68kPaths.root,
        tokenizer=tokenizer,
        transform_datasets=False,
        collation_strategy="sequence_classification",
        num_workers=0,
        batch_size=3,
        fields=gene2vec_unmasked_fields,
        label_columns=zheng68k_label_columns,
        max_length=16,
        limit_dataset_samples={"train": 12, "dev": 12, "predict": 2},
    )
    pl_data_module.prepare_data()
    pl_data_module.setup()
    helpers.update_label_columns(
        pl_data_module.label_columns, pl_data_module.label_dict
    )
    return pl_data_module


@pytest.fixture(scope="session")
def pl_data_module_zheng68k_multitask(
    gene2vec_fields, zheng68k_label_columns, _zheng68k_transform_default
):
    tokenizer = get_gene2vec_tokenizer()
    pl_data_module = Zheng68kDataModule(
        data_dir=helpers.Zheng68kPaths.root,
        tokenizer=tokenizer,
        transform_datasets=False,
        collation_strategy="multitask",
        mlm=True,
        num_workers=0,
        batch_size=3,
        fields=gene2vec_fields,
        label_columns=zheng68k_label_columns,
        max_length=16,
        limit_dataset_samples={"train": 12, "dev": 12, "predict": 2},
    )
    pl_data_module.prepare_data()
    pl_data_module.setup()
    helpers.update_label_columns(
        pl_data_module.label_columns, pl_data_module.label_dict
    )
    return pl_data_module


@pytest.fixture(scope="session")
def pl_data_module_adamson_weissman_seq_labeling(perturbation_fields_cve):
    tokenizer = get_all_genes_tokenizer()
    pl_data_module_adamson_weissman_seq_labeling = ScperturbDataModule(
        data_dir=helpers.ScperturbPerturbationPaths.root,
        tokenizer=tokenizer,
        transform_datasets=True,
        collation_strategy="sequence_labeling",
        num_workers=0,
        batch_size=3,
        fields=perturbation_fields_cve,
        max_length=128,
        limit_dataset_samples=12,
        sequence_order="sorted",
        limit_genes="tokenizer",
    )
    pl_data_module_adamson_weissman_seq_labeling.prepare_data()
    pl_data_module_adamson_weissman_seq_labeling.setup("fit")
    return pl_data_module_adamson_weissman_seq_labeling


@pytest.fixture(scope="session")
def zheng_seq_cls_ckpt(
    pl_data_module_zheng68k_seq_cls: Zheng68kDataModule,
):
    trainer_config = config.TrainerConfig(
        losses=[
            losses.LossTask(
                losses.LabelSource("celltype"), losses.CrossEntropyObjective()
            ),
        ]
    )
    model_config = config.SCBertConfig(
        fields=pl_data_module_zheng68k_seq_cls.fields,
        label_columns=pl_data_module_zheng68k_seq_cls.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = "epoch={epoch}-step={step}-val_loss={validation/loss:.2f}"

        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=3,
            val_check_interval=3,
            enable_progress_bar=False,
            enable_model_summary=False,
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

        seq_cls_training_module = SequenceClassificationTrainingModule(
            model_config=model_config,
            trainer_config=trainer_config,
            label_dict=pl_data_module_zheng68k_seq_cls.label_dict,
        )
        pl_data_module_zheng68k_seq_cls.tokenizer.save_pretrained(
            tmpdir,
            legacy_format=not pl_data_module_zheng68k_seq_cls.tokenizer.is_fast,
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_zheng68k_seq_cls,
            pl_module=seq_cls_training_module,
            task_config=task_config,
        )

        ckpt_path = task_config.default_root_dir + "/last.ckpt"

        yield ckpt_path

    return


@pytest.fixture(scope="session")
def zheng_mlm_ckpt(pl_zheng_mlm_raw_counts):
    trainer_config = config.TrainerConfig(
        losses=[
            losses.LossTask(
                losses.FieldSource("genes"), losses.CrossEntropyObjective(), metrics=[]
            ),
            losses.LossTask(losses.FieldSource("expressions"), losses.MSEObjective()),
            losses.LossTask(
                losses.FieldSource("expressions"), losses.IsZeroBCEObjective()
            ),
        ]
    )
    model_config = config.SCBertConfig(
        fields=pl_zheng_mlm_raw_counts.fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = "epoch={epoch}-step={step}-val_loss={validation/loss:.2f}"

        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=3,
            val_check_interval=3,
            precision="32",
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
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

        training_module = MLMTrainingModule(
            model_config, trainer_config, pl_zheng_mlm_raw_counts.tokenizer
        )
        pl_zheng_mlm_raw_counts.tokenizer.save_pretrained(
            tmpdir,
            legacy_format=not pl_zheng_mlm_raw_counts.tokenizer.is_fast,
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_data_module=pl_zheng_mlm_raw_counts,
            pl_module=training_module,
            task_config=task_config,
            pl_trainer=pl_trainer,
        )

        ckpt_path = task_config.default_root_dir + "/last.ckpt"

        yield ckpt_path

    return


@pytest.fixture(scope="session")
def zheng_multitask_ckpt(
    pl_data_module_zheng68k_multitask: Zheng68kDataModule,
):
    mlm_losses = helpers.default_mlm_losses_from_fields(
        pl_data_module_zheng68k_multitask.fields
    )
    loss_tasks = mlm_losses + [
        losses.LossTask(
            losses.LabelSource(l.label_column_name), losses.CrossEntropyObjective()
        )
        for l in pl_data_module_zheng68k_multitask.label_columns
    ]
    trainer_config = config.TrainerConfig(losses=loss_tasks)
    model_config = config.SCBertConfig(
        fields=pl_data_module_zheng68k_multitask.fields,
        label_columns=pl_data_module_zheng68k_multitask.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = "epoch={epoch}-step={step}-val_loss={validation/loss:.2f}"

        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=3,
            val_check_interval=3,
            enable_progress_bar=False,
            enable_model_summary=False,
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

        multitask_training_module = MultiTaskTrainingModule(
            model_config,
            trainer_config,
            tokenizer=pl_data_module_zheng68k_multitask.tokenizer,
            label_dict=pl_data_module_zheng68k_multitask.label_dict,
        )
        pl_data_module_zheng68k_multitask.tokenizer.save_pretrained(
            tmpdir,
            legacy_format=not pl_data_module_zheng68k_multitask.tokenizer.is_fast,
            # filename_prefix="multifield",
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_zheng68k_multitask,
            pl_module=multitask_training_module,
            task_config=task_config,
        )

        ckpt_path = task_config.default_root_dir + "/last.ckpt"

        yield ckpt_path

    return


@pytest.fixture(scope="session")
def zheng_mlm_rda_ckpt(
    pl_zheng_mlm_raw_counts: Zheng68kDataModule,
    all_genes_fields_with_rda_regression_masking: list[config.FieldInfo],
):
    dm = Zheng68kDataModule(
        data_dir=helpers.Zheng68kPaths.root,
        processed_name=helpers.Zheng68kPaths.raw_counts_name,
        label_columns=pl_zheng_mlm_raw_counts.label_columns,
        transform_datasets=False,
        tokenizer=pl_zheng_mlm_raw_counts.tokenizer,
        batch_size=2,
        fields=all_genes_fields_with_rda_regression_masking,
        limit_dataset_samples=8,
        mlm=True,
        collation_strategy="language_modeling",
        rda_transform="downsample",
        max_length=20,
        pad_to_multiple_of=2,
    )

    trainer_config = config.TrainerConfig(
        losses=[
            losses.LossTask(losses.FieldSource("expressions"), losses.MSEObjective()),
            losses.LossTask(
                losses.FieldSource("expressions"), losses.IsZeroBCEObjective()
            ),
        ]
    )
    model_config = config.SCBertConfig(
        fields=dm.fields,
        label_columns=dm.label_columns,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = "epoch={epoch}-step={step}-val_loss={validation/loss:.2f}"

        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir,
            accelerator="cpu",
            max_epochs=1,
            max_steps=3,
            val_check_interval=3,
            enable_progress_bar=False,
            enable_model_summary=False,
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

        pl_module = MLMTrainingModule(model_config, trainer_config, dm.tokenizer)
        dm.tokenizer.save_pretrained(
            tmpdir,
            legacy_format=not dm.tokenizer.is_fast,
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=dm,
            pl_module=pl_module,
            task_config=task_config,
        )

        ckpt_path = task_config.default_root_dir + "/last.ckpt"

        yield ckpt_path

    return


@pytest.fixture(scope="session")
def sciplex3_seqcls_mt_model_and_ckpt(gene2vec_fields):
    label_columns = [
        config.LabelColumnInfo(
            label_column_name="target", classifier_depth=2, task_group="drug"
        ),
        config.LabelColumnInfo(label_column_name="cell_type", classifier_depth=1),
    ]
    dm = sciplex3.SciPlex3DataModule(
        dataset_kwargs={
            "data_dir": helpers.SciPlex3Paths.root,
            "split_column": "split_random",
        },
        tokenizer=load_tokenizer("gene2vec"),
        fields=gene2vec_fields,
        label_columns=label_columns,
        batch_size=3,
        max_length=8,
        pad_to_multiple_of=2,
        collation_strategy="sequence_classification",
    )
    dm.setup("fit")
    helpers.update_label_columns(label_columns, dm.label_dict)
    cls_label_name = label_columns[0].label_column_name

    model_config_factory = config.SCBertConfig
    fields = gene2vec_fields

    model_config = model_config_factory(
        fields=fields,
        label_columns=label_columns,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        hidden_size=16,
    )

    tasks = [
        losses.LossTask(losses.LabelSource("target"), losses.CrossEntropyObjective()),
        losses.LossTask(
            losses.LabelSource("cell_type"), losses.CrossEntropyObjective()
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = helpers.get_test_task_config(tmpdir)
        mt_pl_module = SequenceClassificationTrainingModule(
            model_config,
            config.TrainerConfig(losses=tasks),
            label_dict=dm.label_dict,
        )

        dm.tokenizer.save_pretrained(
            str(task_config.default_root_dir),
            legacy_format=not dm.tokenizer.is_fast,
            filename_prefix="multifield",
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=dm,
            pl_module=mt_pl_module,
            task_config=task_config,
        )

        ckpt_path = task_config.default_root_dir + "/last.ckpt"

        trained_model = mt_pl_module.model
        yield trained_model, ckpt_path
    return


@pytest.fixture(scope="session")
def sciplex3_mt_model_and_ckpt(gene2vec_fields):
    label_columns = [
        config.LabelColumnInfo(label_column_name="target"),
        config.LabelColumnInfo(label_column_name="cell_type"),
    ]
    dm = sciplex3.SciPlex3DataModule(
        dataset_kwargs={
            "data_dir": helpers.SciPlex3Paths.root,
            "split_column": "split_random",
        },
        tokenizer=load_tokenizer("gene2vec"),
        fields=gene2vec_fields,
        label_columns=label_columns,
        batch_size=3,
        max_length=8,
        pad_to_multiple_of=2,
        collation_strategy="multitask",
        mlm=True,  # Enable MLM to include field labels for expressions
    )
    dm.setup("fit")
    helpers.update_label_columns(label_columns, dm.label_dict)
    cls_label_name = label_columns[0].label_column_name

    model_config_factory = config.SCBertConfig
    fields = gene2vec_fields

    model_config = model_config_factory(
        fields=fields,
        label_columns=label_columns,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        hidden_size=16,
    )

    tasks = [
        losses.LossTask(losses.LabelSource("target"), losses.CrossEntropyObjective()),
        losses.LossTask(
            losses.LabelSource("cell_type"), losses.CrossEntropyObjective()
        ),
        losses.LossTask(
            losses.FieldSource("expressions"), losses.CrossEntropyObjective()
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = helpers.get_test_task_config(tmpdir)
        mt_pl_module = MultiTaskTrainingModule(
            model_config,
            config.TrainerConfig(losses=tasks),
            label_dict=dm.label_dict,
        )

        dm.tokenizer.save_pretrained(
            str(task_config.default_root_dir),
            legacy_format=not dm.tokenizer.is_fast,
            filename_prefix="multifield",
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=dm,
            pl_module=mt_pl_module,
            task_config=task_config,
        )

        ckpt_path = task_config.default_root_dir + "/last.ckpt"

        trained_model = mt_pl_module.model
        yield trained_model, ckpt_path
    return


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    clean_processed_data()


@pytest.fixture(scope="session")
def zheng_dataset_kwargs_after_transform_without_labels(
    pl_data_module_zheng68k_seq_cls,
):
    ds_kwargs = {
        "processed_data_source": pl_data_module_zheng68k_seq_cls.processed_data_file,
        "label_dict_path": Zheng68kPaths.label_dict_path,
        "split_column_name": "split_stratified_celltype",
    }
    return ds_kwargs


@pytest.fixture(scope="session")
def zheng_dataset_kwargs_after_transform(pl_data_module_zheng68k_seq_cls):
    ds_kwargs = {
        "processed_data_source": pl_data_module_zheng68k_seq_cls.processed_data_file,
        "label_dict_path": Zheng68kPaths.label_dict_path,
        "label_columns": ["celltype", "cell_type_ontology_term_id"],
        "stratifying_label": "celltype",
        "expose_zeros": "all",
    }
    return ds_kwargs


@pytest.fixture()
def modular_tokenizer():
    from bmfm_core.data.tokenizers.modular_tokenizer import (
        pretrained as pretrained_tokenizers,
    )
    from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

    base_tokenizer_name = "bmfm_extended_modular_tokenizer"
    tokenizer_path = os.path.join(
        pretrained_tokenizers.get_dir_path(), base_tokenizer_name
    )
    tokenizer_op = ModularTokenizerOp(
        tokenizer_path=tokenizer_path,
        pad_token="<PAD>",
        validate_ends_with_eos=None,
    )
    return tokenizer_op


@pytest.fixture(scope="session")
def single_perts_df():
    gene_tokens = [
        "token1",
        "token2",
        "token3",
        "token4",
        "token5",
        "token6",
        "token7",
        "token8",
        "token9",
        "token10",
        "Control",
    ]
    metadata = pd.DataFrame()
    metadata["perturbation"] = np.random.choice(gene_tokens, 1000)
    return metadata


@pytest.fixture(scope="session")
def combo_perts_df():
    gene_tokens = [
        "token1",
        "token2",
        "token3",
        "token4",
        "token5",
        "token6",
        "token7",
        "token8",
        "token9",
        "token10",
        "Control",
    ]
    metadata = pd.DataFrame()
    combo_perts = []
    for _ in range(500):
        combo_pert = random.sample(gene_tokens[0:5], 2)
        if "Control" in combo_pert:
            combo_perts.append("Control")
        else:
            combo_perts.append("_".join(combo_pert))
    for _ in range(500):
        combo_pert = random.sample(gene_tokens[3:], 2)
        if "Control" in combo_pert:
            combo_perts.append("Control")
        else:
            combo_perts.append("_".join(combo_pert))
    metadata["perturbation"] = combo_perts
    return metadata


@pytest.fixture(scope="session")
def single_and_combo_perts_df(single_perts_df, combo_perts_df):
    metadata = pd.concat([single_perts_df, combo_perts_df])
    return metadata


@pytest.fixture(scope="session")
def cellxgene_label_columns():
    cellxgene_label_dicts = [
        {
            "label_column_name": "cell_type",
        },
        {
            "label_column_name": "tissue",
        },
    ]
    cellxgene_label_columns = [
        config.LabelColumnInfo(**ld) for ld in cellxgene_label_dicts
    ]
    return cellxgene_label_columns


@pytest.fixture(scope="session")
def pl_module_cellxgene_soma(cellxgene_label_columns):
    tokenizer = load_tokenizer()
    dm = CellXGeneSOMADataModule(
        dataset_kwargs={
            "uri": None,
            "shuffle": False,
        },
        tokenizer=tokenizer,
        fields=default_fields(),
        label_columns=cellxgene_label_columns,
        num_workers=0,
        collation_strategy="sequence_classification",
    )
    dm.setup("fit")
    return dm


@pytest.fixture(scope="session")
def dnaseq_fields():
    dnaseq_field_dicts = [
        {
            "field_name": "dna_chunks",
            "pretrained_embedding": None,
            "is_masked": False,
            "vocab_update_strategy": "static",
        },
    ]
    dnaseq_fields = [config.FieldInfo(**fd) for fd in dnaseq_field_dicts]
    tokenizer = load_tokenizer("ref2vec")
    for field in dnaseq_fields:
        field.update_vocab_size(tokenizer)
    return dnaseq_fields


@pytest.fixture(scope="session")
def pl_data_module_dnaseq_core_promoter(dnaseq_fields):
    promoter_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqCorePromoterPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqCorePromoterPaths.label_dict_path,
        "num_workers": 0,
    }
    tokenizer = load_tokenizer("ref2vec")
    label_columns = [
        config.LabelColumnInfo(label_column_name="label", n_unique_values=2)
    ]
    pl_data_module_dnaseq = DNASeqPromoterDataModule(
        tokenizer=tokenizer,
        batch_size=8,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=promoter_dataset_kwargs,
    )

    pl_data_module_dnaseq.setup("fit")
    return pl_data_module_dnaseq


@pytest.fixture(scope="session")
def pl_data_module_dnaseq_promoter(dnaseq_fields):
    promoter_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqPromoterPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqPromoterPaths.label_dict_path,
        "num_workers": 0,
    }
    tokenizer = load_tokenizer("ref2vec")
    label_columns = [config.LabelColumnInfo(label_column_name="promoter_presence")]
    pl_data_module_dnaseq = DNASeqPromoterDataModule(
        tokenizer=tokenizer,
        batch_size=5,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=promoter_dataset_kwargs,
    )

    pl_data_module_dnaseq.setup("fit")
    helpers.update_label_columns(label_columns, pl_data_module_dnaseq.label_dict)
    return pl_data_module_dnaseq


@pytest.fixture(scope="session")
def pl_data_module_dnaseq_lenti_mpra(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqMPRAPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqMPRAPaths.label_dict_path,
    }
    label_columns = [
        config.LabelColumnInfo(label_column_name="mean_value", is_regression_label=True)
    ]
    tokenizer = load_tokenizer("ref2vec")

    for field in dnaseq_fields:
        field.update_vocab_size(tokenizer)

    pl_data_module_dnaseq = DNASeqMPRADataModule(
        tokenizer=tokenizer,
        batch_size=5,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )

    pl_data_module_dnaseq.setup("fit")
    helpers.update_label_columns(label_columns, pl_data_module_dnaseq.label_dict)
    return pl_data_module_dnaseq


@pytest.fixture(scope="session")
def pl_data_module_dnaseq_splice(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqSpliceSitePaths.processed_data_source,
        "label_dict_path": helpers.DNASeqSpliceSitePaths.label_dict_path,
    }
    tokenizer = load_tokenizer("ref2vec")

    for field in dnaseq_fields:
        field.update_vocab_size(tokenizer)

    label_columns = [config.LabelColumnInfo(label_column_name="label")]
    pl_data_module_dnaseq = DNASeqSpliceSiteDataModule(
        tokenizer=tokenizer,
        batch_size=5,
        fields=dnaseq_fields,
        label_columns=label_columns,
        num_workers=0,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )

    pl_data_module_dnaseq.setup("fit")
    helpers.update_label_columns(label_columns, pl_data_module_dnaseq.label_dict)
    return pl_data_module_dnaseq


@pytest.fixture(scope="session")
def pl_data_module_dnaseq_covid(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqCovidPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqCovidPaths.label_dict_path,
    }

    tokenizer = load_tokenizer("ref2vec")

    for field in dnaseq_fields:
        field.update_vocab_size(tokenizer)
    label_columns = [config.LabelColumnInfo(label_column_name="label")]
    pl_data_module_dnaseq = DNASeqCovidDataModule(
        tokenizer=tokenizer,
        batch_size=5,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        num_workers=0,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )

    pl_data_module_dnaseq.setup("fit")
    helpers.update_label_columns(label_columns, pl_data_module_dnaseq.label_dict)
    return pl_data_module_dnaseq


@pytest.fixture(scope="session")
def pl_data_module_dnaseq_chromatin(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqChromatinProfilePaths.processed_data_source,
        "label_dict_path": helpers.DNASeqChromatinProfilePaths.label_dict_path,
    }
    label_columns = [
        config.LabelColumnInfo(label_column_name="dnase_0"),
        config.LabelColumnInfo(label_column_name="dnase_1"),
        config.LabelColumnInfo(label_column_name="dnase_2"),
    ]

    tokenizer = load_tokenizer("ref2vec")

    for field in dnaseq_fields:
        field.update_vocab_size(tokenizer)

    pl_data_module_dnaseq = DNASeqChromatinProfileDataModule(
        tokenizer=tokenizer,
        batch_size=5,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        num_workers=0,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )

    pl_data_module_dnaseq.setup("fit")
    helpers.update_label_columns(label_columns, pl_data_module_dnaseq.label_dict)
    return pl_data_module_dnaseq


@pytest.fixture(scope="session")
def pl_data_module_dnaseq_drosophila_enhancer(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqDrosophilaEnhancerPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqDrosophilaEnhancerPaths.label_dict_path,
    }

    tokenizer = load_tokenizer("ref2vec")

    for field in dnaseq_fields:
        field.update_vocab_size(tokenizer)

    label_columns = [
        config.LabelColumnInfo(
            label_column_name="Dev_log2_enrichment", is_regression_label=True
        )
    ]
    pl_data_module_dnaseq = DNASeqDrosophilaEnhancerDataModule(
        tokenizer=tokenizer,
        batch_size=5,
        fields=dnaseq_fields,
        label_columns=label_columns,
        num_workers=0,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )

    pl_data_module_dnaseq.setup("fit")
    return pl_data_module_dnaseq


@pytest.fixture(scope="session")
def pl_data_module_dnaseq_epigenetic_marks(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqEpigeneticMarksPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqEpigeneticMarksPaths.label_dict_path,
    }

    tokenizer = load_tokenizer("ref2vec")
    for field in dnaseq_fields:
        field.update_vocab_size(tokenizer)

    label_columns = [config.LabelColumnInfo(label_column_name="label")]
    pl_data_module_dnaseq = DNASeqEpigeneticMarksDataModule(
        tokenizer=tokenizer,
        batch_size=5,
        fields=dnaseq_fields,
        label_columns=label_columns,
        num_workers=0,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )

    pl_data_module_dnaseq.setup("fit")
    helpers.update_label_columns(label_columns, pl_data_module_dnaseq.label_dict)
    return pl_data_module_dnaseq


@pytest.fixture(scope="session")
def pl_data_module_dnaseq_transcription_factor(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqTranscriptionFactorPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqTranscriptionFactorPaths.label_dict_path,
    }

    tokenizer = load_tokenizer("ref2vec")
    for field in dnaseq_fields:
        field.update_vocab_size(tokenizer)

    label_columns = [config.LabelColumnInfo(label_column_name="label")]
    pl_data_module_dnaseq = DNASeqTranscriptionFactorDataModule(
        tokenizer=tokenizer,
        batch_size=5,
        fields=dnaseq_fields,
        label_columns=label_columns,
        num_workers=0,
        collation_strategy="sequence_classification",
        max_length=512,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )

    pl_data_module_dnaseq.setup("fit")
    helpers.update_label_columns(label_columns, pl_data_module_dnaseq.label_dict)

    return pl_data_module_dnaseq


@pytest.fixture()
def mock_clearml_logger(monkeypatch):
    """Mock ClearML logger to print calls to console."""

    def logger_method(*args, **kwargs):
        logger.info(f"ClearML Mock - {method_name}: {args}, {kwargs}")

    mock_logger = MagicMock()
    mock_logger.current_logger.return_value = mock_logger

    # Dynamically add logging methods to the mock
    for method_name in [
        "report_confusion_matrix",
        "report_histogram",
        "report_single_value",
        "report_table",
    ]:
        mock_logger.__getattr__(method_name).side_effect = logger_method

    monkeypatch.setattr(
        "clearml.logger.Logger.current_logger", mock_logger.current_logger
    )
    return mock_logger


@pytest.fixture(scope="session")
def pl_data_module_dnaseq_snp2trait(dnaseq_fields):
    dnaseq_dataset_kwargs = {
        "processed_data_source": helpers.DNASeqSnp2TraitPaths.processed_data_source,
        "label_dict_path": helpers.DNASeqSnp2TraitPaths.label_dict_path,
    }
    label_columns = [
        config.LabelColumnInfo(label_column_name="trait_0"),
        config.LabelColumnInfo(label_column_name="trait_1"),
        config.LabelColumnInfo(label_column_name="trait_2"),
    ]

    tokenizer = load_tokenizer("ref2vec")

    for field in dnaseq_fields:
        field.update_vocab_size(tokenizer)

    pl_data_module_dnaseq = DNASeqSnp2TraitDataModule(
        tokenizer=tokenizer,
        batch_size=5,
        fields=dnaseq_fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        max_length=512,
        num_workers=0,
        dataset_kwargs=dnaseq_dataset_kwargs,
    )

    pl_data_module_dnaseq.setup("fit")
    helpers.update_label_columns(label_columns, pl_data_module_dnaseq.label_dict)
    return pl_data_module_dnaseq
