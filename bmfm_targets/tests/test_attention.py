import tempfile

import numpy as np
import pytest
import torch

from bmfm_targets import config
from bmfm_targets.config import SCBertConfig
from bmfm_targets.datasets.panglaodb import PanglaoDBDataModule
from bmfm_targets.models.predictive.attentions import (
    SCBertCustomAttention,
    SelfTorchAttention,
)
from bmfm_targets.models.predictive.scbert.modeling_scbert import SCBertAttention
from bmfm_targets.tasks.task_utils import make_trainer_for_task, train
from bmfm_targets.tests.helpers import (
    default_mlm_losses_from_fields,
    get_test_task_config,
)
from bmfm_targets.training.modules import MLMTrainingModule

_attention_pars = [
    {"attention": "torch"},
]


@pytest.fixture(scope="module", params=_attention_pars)
def attention_model_config(gene2vec_fields, request):
    model_config = config.SCBertConfig(
        fields=gene2vec_fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        hidden_size=32,
        **request.param,
    )
    return model_config


def test_HF_and_torch_attention_model_compatibility():
    config = SCBertConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_act="gelu",
        hidden_size=32,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        use_cache=True,
        classifier_dropout=None,
    )
    hf_attention = SCBertAttention(config)
    torch_attention = SCBertCustomAttention(
        config, self_attention=SelfTorchAttention(config)
    )

    torch_attention.load_state_dict(hf_attention.state_dict())
    hf_attention.load_state_dict(torch_attention.state_dict())

    torch_attention.eval()
    hf_attention.eval()

    batch_size = 2
    num_tokens = 100
    hidden_states = torch.rand((batch_size, num_tokens, config.hidden_size))
    copy_hidden_states = torch.clone(hidden_states)

    with torch.no_grad():
        torch_vals = [i.detach().cpu().numpy() for i in torch_attention(hidden_states)]
        hf_vals = [i.detach().cpu().numpy() for i in hf_attention(copy_hidden_states)]

    for torch_val, hf_val in zip(torch_vals, hf_vals):
        np.testing.assert_allclose(torch_val, hf_val, atol=1e-3)


def test_torch_attention_train_scbert(
    pl_data_module_panglao: PanglaoDBDataModule,
    attention_model_config,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = get_test_task_config(tmpdir)
        trainer_config = config.TrainerConfig(
            losses=default_mlm_losses_from_fields(pl_data_module_panglao.fields)
        )
        mlm_training_module = MLMTrainingModule(
            attention_model_config, trainer_config, pl_data_module_panglao.tokenizer
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=pl_data_module_panglao,
            pl_module=mlm_training_module,
            task_config=task_config,
        )
