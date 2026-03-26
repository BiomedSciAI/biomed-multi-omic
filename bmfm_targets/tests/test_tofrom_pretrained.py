import tempfile

import numpy as np
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification

from bmfm_targets import config
from bmfm_targets.models import register_configs_and_models
from bmfm_targets.models.predictive import scbert, scnystromformer, scperformer


def test_can_load_model_autoconfig_scbert(gene2vec_fields):
    register_configs_and_models()
    with tempfile.TemporaryDirectory() as save_path:
        verify_automodel_from_pretrained_works(
            gene2vec_fields,
            config.SCBertConfig,
            scbert.SCBertForMultiTaskModeling,
            save_path,
        )


def test_can_load_model_autoconfig_scperformer(gene2vec_fields):
    register_configs_and_models()
    with tempfile.TemporaryDirectory() as save_path:
        verify_automodel_from_pretrained_works(
            gene2vec_fields,
            config.SCPerformerConfig,
            scperformer.SCPerformerForMultiTaskModeling,
            save_path,
        )


def test_can_load_model_autoconfig_scnystromformer(gene2vec_fields):
    register_configs_and_models()
    with tempfile.TemporaryDirectory() as save_path:
        verify_automodel_from_pretrained_works(
            gene2vec_fields,
            config.SCNystromformerConfig,
            scnystromformer.SCNystromformerForMultiTaskModeling,
            save_path,
        )


def verify_automodel_from_pretrained_works(
    gene2vec_fields, config_factory, multitask_factory, save_path
):
    """
    Verify that AutoModel can load multitask models for both MLM and SeqCls.

    After refactoring, there's only one model type (ForMultiTaskModeling) but
    AutoModelForMaskedLM and AutoModelForSequenceClassification both load it.
    """
    model_type = config_factory.model_type
    model_config = config_factory(
        fields=gene2vec_fields,
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
    )

    # Create and save a multitask model
    model = multitask_factory(model_config)
    model.save_pretrained(save_path)

    dummy_label_columns = [
        config.LabelColumnInfo(label_column_name="dummy", n_unique_values=10)
    ]

    # Both AutoModel types should load the same ForMultiTaskModeling class
    mlm = AutoModelForMaskedLM.from_pretrained(save_path)
    seq_cls = AutoModelForSequenceClassification.from_pretrained(
        save_path, label_columns=dummy_label_columns
    )

    # Both should be instances of the multitask model
    assert isinstance(seq_cls, multitask_factory)
    assert isinstance(mlm, multitask_factory)

    # Verify embeddings are preserved across loading
    gene_embeddings_seq_cls = (
        getattr(seq_cls, model_type).embeddings.genes_embeddings.weight.detach().numpy()
    )
    gene_embeddings_mlm = (
        getattr(mlm, model_type).embeddings.genes_embeddings.weight.detach().numpy()
    )
    gene_embeddings_original = (
        getattr(model, model_type).embeddings.genes_embeddings.weight.detach().numpy()
    )

    np.testing.assert_allclose(gene_embeddings_mlm, gene_embeddings_original)
    np.testing.assert_allclose(gene_embeddings_seq_cls, gene_embeddings_original)


# Made with Bob
