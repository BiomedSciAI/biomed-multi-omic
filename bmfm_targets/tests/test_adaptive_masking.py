import tempfile

import pytest

from bmfm_targets import config
from bmfm_targets.datasets import zheng68k
from bmfm_targets.tasks.task_utils import make_trainer_for_task, train
from bmfm_targets.tests.helpers import Zheng68kPaths, get_test_task_config
from bmfm_targets.tokenization import load_tokenizer
from bmfm_targets.training.losses import (
    CrossEntropyObjective,
    FieldSource,
    IsZeroBCEObjective,
    LossTask,
    MSEObjective,
    WCEDFieldSource,
)
from bmfm_targets.training.masking.adaptive_strategy import (
    AdaptiveMaskingStrategy,
    AdaptiveWCEDMasker,
)
from bmfm_targets.training.modules import MLMTrainingModule


def test_can_update_masking_probs(pl_zheng_mlm_raw_counts):
    dm = pl_zheng_mlm_raw_counts
    ms = AdaptiveMaskingStrategy(tokenizer=dm.tokenizer)
    mfis = dm.train_dataset[:5]
    batch = dm.collate_fn.tokenize_batch(mfis)
    masking_probs = ms.get_mask_probs(batch)

    genes_to_upweight = mfis[0]["genes"][:5]
    genes_to_downweight = mfis[0]["genes"][5:10]

    updated_token_probs = {}
    for u, d in zip(genes_to_upweight, genes_to_downweight):
        updated_token_probs[u] = 2.0
        updated_token_probs[d] = 0.5

    ms.update_token_masking_probs(updated_token_probs)
    updated_masking_probs = ms.get_mask_probs(batch)
    assert not (updated_masking_probs == masking_probs).all()


@pytest.mark.skip(reason="sps integration fails, requires additional review")
def test_updatable_token_masking_prob_masking_strategy_inside_masker(
    pl_zheng_mlm_raw_counts,
):
    dm = zheng68k.Zheng68kDataModule(
        fields=pl_zheng_mlm_raw_counts.fields,
        tokenizer=pl_zheng_mlm_raw_counts.tokenizer,
        data_dir=pl_zheng_mlm_raw_counts.data_dir,
        processed_name=Zheng68kPaths.raw_counts_name,
        transform_kwargs={"transforms": []},
        masking_strategy=AdaptiveMaskingStrategy(
            tokenizer=pl_zheng_mlm_raw_counts.tokenizer
        ),
        label_columns=pl_zheng_mlm_raw_counts.label_columns,
        transform_datasets=False,
        batch_size=4,
        limit_dataset_samples={"train": 32, "dev": 256},
        mlm=True,
        collation_strategy="language_modeling",
        rda_transform=2000,
        sequence_order="random",
        max_length=64,
        pad_to_multiple_of=2,
        num_workers=2,
    )
    dm.prepare_data()
    dm.setup("fit")

    model_config = config.SCBertConfig(
        fields=dm.fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
        pad_token_id=2,
    )
    trainer_config = config.TrainerConfig(
        losses=[
            LossTask(
                source=FieldSource(field_name="genes"),
                objective=CrossEntropyObjective(),
                weight=1,
                metrics=[],
            ),
            LossTask(
                source=FieldSource(field_name="expressions"),
                objective=MSEObjective(ignore_zero=True),
                weight=1,
            ),
            LossTask(
                source=FieldSource(field_name="expressions"),
                objective=IsZeroBCEObjective(),
                weight=1,
            ),
        ],
        batch_prediction_behavior="track",
    )
    masking_strategy = dm.masking_strategy
    assert masking_strategy is not None
    p = masking_strategy.token_masking_probs
    assert all(p == 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = get_test_task_config(tmpdir)
        task_config.max_epochs = 5
        task_config.max_steps = -1
        pl_trainer = make_trainer_for_task(task_config)
        pl_module = MLMTrainingModule(model_config, trainer_config, dm.tokenizer)
        train(
            pl_trainer, pl_data_module=dm, pl_module=pl_module, task_config=task_config
        )

    masking_strategy = dm.masking_strategy
    p = masking_strategy.token_masking_probs
    assert any(p < 1)


def test_masking_strategy_can_be_turned_off_in_val(pl_zheng_mlm_raw_counts):
    dm = zheng68k.Zheng68kDataModule(
        fields=pl_zheng_mlm_raw_counts.fields,
        tokenizer=pl_zheng_mlm_raw_counts.tokenizer,
        data_dir=pl_zheng_mlm_raw_counts.data_dir,
        processed_name=Zheng68kPaths.raw_counts_name,
        transform_kwargs={"transforms": []},
        masking_strategy=AdaptiveMaskingStrategy(
            tokenizer=pl_zheng_mlm_raw_counts.tokenizer, use_for_validation=False
        ),
        label_columns=pl_zheng_mlm_raw_counts.label_columns,
        transform_datasets=False,
        batch_size=4,
        limit_dataset_samples={"train": 32, "dev": 256},
        mlm=True,
        collation_strategy="language_modeling",
        rda_transform=2000,
        sequence_order="random",
        max_length=64,
        pad_to_multiple_of=2,
        num_workers=2,
    )
    dm.prepare_data()
    dm.setup("fit")
    val_dl = dm.val_dataloader()
    assert val_dl.collate_fn.masker.masking_strategy is None

    train_dl = dm.train_dataloader()
    assert train_dl.collate_fn.masker.masking_strategy is not None


def test_wced_adaptive_masking(
    pl_zheng_mlm_raw_counts,
):
    fields = [
        config.FieldInfo("genes"),
        config.FieldInfo(
            "expressions",
            decode_modes={
                "wced": {
                    "vocab_field": "genes",
                    "logit_outputs": ["mse", "is_zero_bce"],
                }
            },
        ),
    ]

    tokenizer = load_tokenizer("protein_coding")
    for field in fields:
        field.update_vocab_size(tokenizer)
    dm = zheng68k.Zheng68kDataModule(
        fields=fields,
        tokenizer=pl_zheng_mlm_raw_counts.tokenizer,
        data_dir=pl_zheng_mlm_raw_counts.data_dir,
        processed_name=Zheng68kPaths.raw_counts_name,
        transform_kwargs={"transforms": []},
        masking_strategy=AdaptiveWCEDMasker(
            tokenizer=pl_zheng_mlm_raw_counts.tokenizer
        ),
        label_columns=pl_zheng_mlm_raw_counts.label_columns,
        transform_datasets=False,
        batch_size=4,
        limit_dataset_samples={"train": 32, "dev": 256},
        mlm=True,
        sequence_dropout_factor=0.3,
        collation_strategy="language_modeling",
        rda_transform=2000,
        sequence_order="random",
        max_length=64,
        pad_to_multiple_of=2,
    )
    dm.prepare_data()
    dm.setup("fit")

    model_config = config.SCBertConfig(
        fields=dm.fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
        pad_token_id=2,
    )
    trainer_config = config.TrainerConfig(
        losses=[
            LossTask(
                source=WCEDFieldSource(
                    field_name="expressions",
                    wced_target="non_input_genes",
                ),
                objective=MSEObjective(),
            ),
            LossTask(
                source=WCEDFieldSource(
                    field_name="expressions",
                    wced_target="non_input_genes",
                ),
                objective=IsZeroBCEObjective(),
            ),
            LossTask(
                source=WCEDFieldSource(
                    field_name="expressions",
                    wced_target="input_genes",
                ),
                objective=MSEObjective(),
            ),
            LossTask(
                source=WCEDFieldSource(
                    field_name="expressions",
                    wced_target="input_genes",
                ),
                objective=IsZeroBCEObjective(),
            ),
        ],
        batch_prediction_behavior="track",
    )
    masker = dm.masker
    assert masker is not None
    p = masker.token_masking_probs
    assert all(p == 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = get_test_task_config(tmpdir)
        pl_trainer = make_trainer_for_task(task_config)
        pl_module = MLMTrainingModule(model_config, trainer_config, dm.tokenizer)
        train(
            pl_trainer, pl_data_module=dm, pl_module=pl_module, task_config=task_config
        )

    masker = dm.masker
    p = masker.selective_dropout_weights
    assert any(p < 1)


def test_wced_adaptive_masking_can_be_turned_off_in_val(
    pl_zheng_mlm_raw_counts,
):
    fields = [
        config.FieldInfo("genes"),
        config.FieldInfo(
            "expressions",
            decode_modes={
                "wced": {
                    "vocab_field": "genes",
                    "logit_outputs": ["mse", "is_zero_bce"],
                }
            },
        ),
    ]

    tokenizer = load_tokenizer("protein_coding")
    for field in fields:
        field.update_vocab_size(tokenizer)
    dm = zheng68k.Zheng68kDataModule(
        fields=fields,
        tokenizer=pl_zheng_mlm_raw_counts.tokenizer,
        data_dir=pl_zheng_mlm_raw_counts.data_dir,
        processed_name=Zheng68kPaths.raw_counts_name,
        transform_kwargs={"transforms": []},
        masking_strategy=AdaptiveWCEDMasker(
            tokenizer=pl_zheng_mlm_raw_counts.tokenizer, use_for_validation=False
        ),
        label_columns=pl_zheng_mlm_raw_counts.label_columns,
        transform_datasets=False,
        batch_size=4,
        limit_dataset_samples={"train": 32, "dev": 256},
        mlm=True,
        sequence_dropout_factor=0.3,
        collation_strategy="language_modeling",
        rda_transform=2000,
        sequence_order="random",
        max_length=64,
        pad_to_multiple_of=2,
    )
    dm.prepare_data()
    dm.setup("fit")

    model_config = config.SCBertConfig(
        fields=dm.fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
        pad_token_id=2,
    )
    trainer_config = config.TrainerConfig(
        losses=[
            LossTask(
                source=WCEDFieldSource(
                    field_name="expressions",
                    wced_target="non_input_genes",
                ),
                objective=MSEObjective(),
            ),
            LossTask(
                source=WCEDFieldSource(
                    field_name="expressions",
                    wced_target="non_input_genes",
                ),
                objective=IsZeroBCEObjective(),
            ),
            LossTask(
                source=WCEDFieldSource(
                    field_name="expressions",
                    wced_target="input_genes",
                ),
                objective=MSEObjective(),
            ),
            LossTask(
                source=WCEDFieldSource(
                    field_name="expressions",
                    wced_target="input_genes",
                ),
                objective=IsZeroBCEObjective(),
            ),
        ],
        batch_prediction_behavior="track",
    )
    masker = dm.masker
    assert masker is not None
    p = masker.token_masking_probs
    assert all(p == 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = get_test_task_config(tmpdir)
        pl_trainer = make_trainer_for_task(task_config)
        pl_module = MLMTrainingModule(model_config, trainer_config, dm.tokenizer)
        train(
            pl_trainer, pl_data_module=dm, pl_module=pl_module, task_config=task_config
        )

    train_masker = pl_trainer.train_dataloader.collate_fn.masker
    val_masker = pl_trainer.val_dataloaders.collate_fn.masker
    assert any(train_masker.selective_dropout_weights < 1)
    assert all(val_masker.selective_dropout_weights == 1)
