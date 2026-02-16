import os
import tempfile
from functools import partial

import pytest
import torch

from bmfm_targets import config
from bmfm_targets.config.main_config import get_label_output_size_for_model_config
from bmfm_targets.datasets.cellxgene import CellXGeneDataModule
from bmfm_targets.datasets.panglaodb import PanglaoDBDataModule
from bmfm_targets.datasets.zheng68k import Zheng68kDataModule
from bmfm_targets.models import get_model_from_config
from bmfm_targets.models.predictive import scbert, scnystromformer, scperformer
from bmfm_targets.models.predictive.layers import GradientReversal
from bmfm_targets.tests import helpers
from bmfm_targets.tokenization import get_gene2vec_tokenizer
from bmfm_targets.training.modules import (
    MLMTrainingModule,
    SequenceClassificationTrainingModule,
    get_training_module_class_for_data_module,
)


@pytest.fixture()
def fields():
    return [
        config.FieldInfo(
            field_name="text",
            vocab_size=1000,
            pretrained_embedding=None,
            is_masked=False,
            vocab_update_strategy="static",
        ),
        config.FieldInfo(
            field_name="label",
            vocab_size=10,
            pretrained_embedding=None,
            is_masked=True,
            vocab_update_strategy="static",
            decode_modes={"token_scores": {}},
        ),
    ]


@pytest.fixture()
def label_columns():
    label_dicts = [{"label_column_name": "celltype", "n_unique_values": 10}]
    label_columns = [config.LabelColumnInfo(**ld) for ld in label_dicts]
    return label_columns


@pytest.fixture()
def sc_bert_config(fields, label_columns):
    return config.SCBertConfig(fields=fields, label_columns=label_columns)


@pytest.fixture()
def sc_performer_config(fields, label_columns):
    return config.SCPerformerConfig(fields=fields, label_columns=label_columns)


@pytest.fixture()
def sc_nystromformer_config(fields, label_columns):
    return config.SCNystromformerConfig(fields=fields, label_columns=label_columns)


@pytest.fixture()
def all_model_configs(sc_bert_config, sc_performer_config, sc_nystromformer_config):
    return [sc_bert_config, sc_performer_config, sc_nystromformer_config]


def test_can_dump_config_to_json(all_model_configs):
    for model_config in all_model_configs:
        with tempfile.NamedTemporaryFile("w") as f:
            model_config.to_json_file(f.name)


def test_can_choose_correct_sc_performer_model(sc_performer_config):
    mlm_model = get_model_from_config(sc_performer_config, modeling_strategy="mlm")
    seq_cls_model = get_model_from_config(sc_performer_config, modeling_strategy="mlm")
    seq_label_model = get_model_from_config(
        sc_performer_config, modeling_strategy="sequence_labeling"
    )
    seq_cls_model = get_model_from_config(
        sc_performer_config, modeling_strategy="sequence_classification"
    )
    assert isinstance(mlm_model, scperformer.SCPerformerForMaskedLM)
    assert isinstance(seq_cls_model, scperformer.SCPerformerForSequenceClassification)
    assert isinstance(seq_label_model, scperformer.SCPerformerForSequenceLabeling)


def test_can_choose_correct_scbert_model(sc_bert_config):
    mlm_model = get_model_from_config(sc_bert_config, modeling_strategy="mlm")
    seq_cls_model = get_model_from_config(
        sc_bert_config, modeling_strategy="sequence_classification"
    )
    seq_label_model = get_model_from_config(
        sc_bert_config, modeling_strategy="sequence_labeling"
    )

    assert isinstance(mlm_model, scbert.SCBertForMaskedLM)
    assert isinstance(seq_cls_model, scbert.SCBertForSequenceClassification)
    assert isinstance(seq_label_model, scbert.SCBertForSequenceLabeling)


def test_can_choose_correct_scnystromformer_model(sc_nystromformer_config):
    mlm_model = get_model_from_config(sc_nystromformer_config, modeling_strategy="mlm")
    seq_cls_model = get_model_from_config(
        sc_nystromformer_config, modeling_strategy="sequence_classification"
    )
    seq_label_model = get_model_from_config(
        sc_nystromformer_config, modeling_strategy="sequence_labeling"
    )

    assert isinstance(mlm_model, scnystromformer.SCNystromformerForMaskedLM)
    assert isinstance(
        seq_cls_model, scnystromformer.SCNystromformerForSequenceClassification
    )
    assert isinstance(
        seq_label_model, scnystromformer.SCNystromformerForSequenceLabeling
    )


def test_can_choose_training_module_for_data_module(fields, label_columns):
    mlm_data_module = Zheng68kDataModule(
        tokenizer=get_gene2vec_tokenizer(),
        fields=fields,
        collation_strategy="language_modeling",
        mlm=True,
        transform_datasets=False,
    )

    classification_data_module = Zheng68kDataModule(
        tokenizer=get_gene2vec_tokenizer(),
        fields=fields,
        label_columns=label_columns,
        collation_strategy="sequence_classification",
        transform_datasets=False,
    )
    mlm_training_module = get_training_module_class_for_data_module(mlm_data_module)
    assert mlm_training_module == MLMTrainingModule
    classification_training_module = get_training_module_class_for_data_module(
        classification_data_module
    )
    assert classification_training_module == SequenceClassificationTrainingModule


def test_can_derive_output_size_scbert_config(
    fields, all_model_configs, pl_data_module_zheng68k_seq_cls
):
    for model_config in all_model_configs:
        check_for_scbert_config(
            fields, model_config.__class__, pl_data_module_zheng68k_seq_cls
        )


def check_for_scbert_config(
    fields, model_config_class, pl_data_module_zheng68k_seq_cls
):
    partial_model_config = partial(
        model_config_class,
        fields=fields,
        label_columns=pl_data_module_zheng68k_seq_cls.label_columns,
    )()
    output_size = get_label_output_size_for_model_config(
        pl_data_module_zheng68k_seq_cls, partial_model_config
    )
    assert output_size == 11

    panglao_data_module = PanglaoDBDataModule(
        dataset_kwargs={},
        tokenizer=get_gene2vec_tokenizer(),
        fields=fields,
        label_columns=None,
    )
    partial_model_config = partial(
        config.SCBertConfig, fields=fields, label_columns=None
    )()
    output_size = get_label_output_size_for_model_config(
        panglao_data_module, partial_model_config
    )
    assert output_size is None

    dummy_label_columns = [
        config.LabelColumnInfo(label_column_name="dummy_label", n_unique_values=1337)
    ]
    partial_model_config = partial(
        config.SCBertConfig, fields=fields, label_columns=dummy_label_columns
    )()
    output_size = get_label_output_size_for_model_config(
        pl_data_module_zheng68k_seq_cls, partial_model_config
    )
    assert output_size == 1337


def test_can_derive_num_label_columns_with_downsampled_dataset(
    fields, cellxgene_label_columns
):
    # Initialize tokenizer and configurations
    tokenizer = get_gene2vec_tokenizer()
    dataset_kwargs = {
        "label_dict_path": helpers.CellXGenePaths.label_dict_path,
    }
    transform_kwargs = {
        "split_weights": {"train": 0.4, "dev": 0.3, "test": 0.3},
    }

    # Create the full dataset data module
    dm_full = CellXGeneDataModule(
        data_dir=helpers.CellXGenePaths.root,
        dataset_kwargs=dataset_kwargs,
        tokenizer=tokenizer,
        fields=fields,
        label_columns=cellxgene_label_columns,
        transform_kwargs=transform_kwargs,
        mlm=False,
        transform_datasets=True,
    )
    dm_full.prepare_data()
    dm_full.setup()
    helpers.update_label_columns(dm_full.label_columns, dm_full.label_dict)

    partial_model_config = partial(
        config.SCBertConfig, fields=fields, label_columns=dm_full.label_columns
    )()
    output_size_full = get_label_output_size_for_model_config(
        dm_full, partial_model_config
    )

    if os.path.exists(helpers.CellXGenePaths.label_dict_path):
        os.remove(helpers.CellXGenePaths.label_dict_path)

    post_transform_dataset_kwargs = dm_full.dataset_kwargs.copy()
    post_transform_dataset_kwargs["processed_data_source"] = dm_full.processed_data_file
    dm_downsample = CellXGeneDataModule(
        dataset_kwargs=post_transform_dataset_kwargs,
        tokenizer=tokenizer,
        fields=fields,
        label_columns=cellxgene_label_columns,
        mlm=False,
        limit_dataset_samples=30,
        transform_datasets=False,
    )
    dm_downsample.prepare_data()
    dm_downsample.setup()
    helpers.update_label_columns(dm_downsample.label_columns, dm_downsample.label_dict)
    partial_model_config = partial(
        config.SCBertConfig, fields=fields, label_columns=dm_downsample.label_columns
    )()
    output_size_ds = get_label_output_size_for_model_config(
        dm_downsample, partial_model_config
    )
    assert output_size_ds == output_size_full


def test_instantiate_model_with_continuous_value_encoder():
    vocab_size = 100
    num_special_tokens = 5
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions",
            vocab_size=None,
            is_masked=True,
            num_special_tokens=num_special_tokens,
            tokenization_strategy="continuous_value_encoder",
            decode_modes=["regression"],
            encoder_kwargs={"kind": "mlp_with_special_token_embedding"},
        ),
    ]

    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
    )
    model = scbert.SCBertForMaskedLM(model_config)
    from bmfm_targets.models.predictive.layers import (
        ContinuousValueEncoderWithSpecialTokenEmbeddings,
    )

    assert isinstance(
        model.scbert.embeddings.expressions_embeddings,
        ContinuousValueEncoderWithSpecialTokenEmbeddings,
    )


def test_instantiate_model_with_scale_adapt_encoder():
    vocab_size = 100
    num_special_tokens = 5

    encoder_kwargs = {
        "kind": "scale_adapt",
        "n_sin_basis": 11,
        "shift": 0.0,
        "basis_scale": 0.1,
        "sigmoid_centers": [0.0],
        "sigmoid_orientations": [1.0],
    }

    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions",
            vocab_size=None,
            is_masked=True,
            num_special_tokens=num_special_tokens,
            tokenization_strategy="continuous_value_encoder",
            encoder_kwargs=encoder_kwargs,
            decode_modes=["regression"],
        ),
    ]

    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        fields=fields,
    )
    model = scbert.SCBertForMaskedLM(model_config)
    from bmfm_targets.models.predictive.layers import (
        ScaleAdaptEncoder,
    )

    assert isinstance(
        model.scbert.embeddings.expressions_embeddings,
        ScaleAdaptEncoder,
    )


def test_instantiate_model_with_is_zero_decoder():
    vocab_size = 100
    num_special_tokens = 5
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions",
            vocab_size=None,
            is_masked=True,
            num_special_tokens=num_special_tokens,
            tokenization_strategy="continuous_value_encoder",
            decode_modes=["regression", "is_zero"],
            encoder_kwargs={"kind": "mlp_with_special_token_embedding"},
        ),
    ]

    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=128,
        intermediate_size=32,
        fields=fields,
    )
    model = scbert.SCBertForMaskedLM(model_config)

    # we want 1d output for this, just like regression
    assert (
        model.cls.predictions.decoder.field_decoders.expressions_is_zero.weight.shape[0]
        == 1
    )
    assert (
        model.cls.predictions.decoder.field_decoders.expressions_regression.weight.shape[
            0
        ]
        == 1
    )


def test_instantiate_model_with_zero_as_special_token():
    vocab_size = 100
    num_special_tokens = 5
    fields = [
        config.FieldInfo("genes", vocab_size),
        config.FieldInfo(
            "expressions",
            vocab_size=None,
            is_masked=True,
            num_special_tokens=num_special_tokens,
            tokenization_strategy="continuous_value_encoder",
            decode_modes=["regression"],
            encoder_kwargs={
                "kind": "mlp_with_special_token_embedding",
                "zero_as_special_token": True,
            },
        ),
    ]

    model_config = config.SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=128,
        intermediate_size=32,
        fields=fields,
    )
    model = scbert.SCBertForMaskedLM(model_config)

    # we want 1d output for this, just like regression
    assert (
        model.scbert.embeddings.expressions_embeddings.special_token_embeddings.num_embeddings
        == num_special_tokens + 1
    )

    input_values = torch.tensor(
        [[-4, 0.0, 1.2, 5.6, -3], [-4, 2.3, 0.0, -3, -3], [-4, 2.2, 6.3, 0.0, -3]]
    )
    output = model.scbert.embeddings.expressions_embeddings.forward(input_values)
    assert not (output[0][1] == output[0][-1]).all()
    assert (output[0][1] == output[1][2]).all()


def test_instantiate_gradient_reversal_layer(all_genes_fields):
    cat_grl_coef = 0.1
    reg_grl_coef = 0.2
    label_columns = [
        config.LabelColumnInfo(
            "bad_cat_label",
            n_unique_values=10,
            gradient_reversal_coefficient=cat_grl_coef,
        ),
        config.LabelColumnInfo(
            "bad_reg_label",
            is_regression_label=True,
            gradient_reversal_coefficient=reg_grl_coef,
        ),
        config.LabelColumnInfo("good_cat_label", n_unique_values=10),
        config.LabelColumnInfo("good_reg_label", is_regression_label=True),
    ]
    model_config = config.SCBertConfig(
        fields=all_genes_fields, label_columns=label_columns
    )
    model = get_model_from_config(model_config, modeling_strategy="multitask")

    decoders = model.cls.label_predictions.predictions.label_decoders
    assert isinstance(decoders["bad_cat_label"].decoder[0], GradientReversal)
    assert isinstance(decoders["bad_reg_label"].decoder[0], GradientReversal)

    torch.testing.assert_close(
        decoders["bad_cat_label"].decoder[0].alpha.item(), cat_grl_coef
    )
    torch.testing.assert_close(
        decoders["bad_reg_label"].decoder[0].alpha.item(), reg_grl_coef
    )

    assert isinstance(decoders["good_reg_label"].decoder, torch.nn.Linear)
    assert isinstance(decoders["good_reg_label"].decoder, torch.nn.Linear)
