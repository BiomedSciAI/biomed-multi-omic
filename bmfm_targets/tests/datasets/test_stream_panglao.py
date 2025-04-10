import tempfile

import pytest

import bmfm_targets.config as config
from bmfm_targets.datasets.panglaodb import StreamingPanglaoDBDataModule
from bmfm_targets.tasks.task_utils import make_trainer_for_task, train
from bmfm_targets.training.modules import MLMTrainingModule

from ..helpers import default_mlm_losses_from_fields


@pytest.mark.usefixtures("_convert_rdata_to_litdata")
def test_init(streaming_panglao_parameters):
    datamodule = StreamingPanglaoDBDataModule(**streaming_panglao_parameters)
    datamodule.prepare_data()
    datamodule.setup()
    assert datamodule is not None


@pytest.mark.usefixtures("_convert_rdata_to_litdata")
def test_reading_content(streaming_panglao_parameters):
    datamodule = StreamingPanglaoDBDataModule(**streaming_panglao_parameters)
    datamodule.prepare_data()
    datamodule.setup()
    num_records = 200

    train_dataloader = datamodule.train_dataloader()
    n = 0
    for i in train_dataloader:
        n += i["input_ids"].shape[0]
    assert n == num_records * 2


@pytest.mark.usefixtures("_convert_rdata_to_litdata")
def test_restart(streaming_panglao_parameters):
    datamodule = StreamingPanglaoDBDataModule(**streaming_panglao_parameters)
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()

    num_records = 200

    record = iter(train_dataloader)
    n = 0
    for i in range(num_records // 2):
        batch = next(record)
        n += batch["input_ids"].shape[0]

    assert n == num_records

    state = datamodule.get_train_dataloader_state()
    datamodule = StreamingPanglaoDBDataModule(**streaming_panglao_parameters)
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    train_dataloader.load_state_dict(state)

    for i in train_dataloader:
        n += i["input_ids"].shape[0]

    assert n == num_records * 2


@pytest.mark.usefixtures("_convert_rdata_to_litdata")
def test_restart_litdata_failure(streaming_panglao_parameters):
    streaming_panglao_parameters["num_workers"] = 5
    streaming_panglao_parameters["batch_size"] = 2
    datamodule = StreamingPanglaoDBDataModule(**streaming_panglao_parameters)
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()

    num_records = 200

    record = iter(train_dataloader)
    n = 0
    for i in range(num_records - 1):
        batch = next(record)
        n += batch["input_ids"].shape[0]

    state = datamodule.get_train_dataloader_state()
    datamodule = StreamingPanglaoDBDataModule(**streaming_panglao_parameters)
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    train_dataloader.load_state_dict(state)

    for i in train_dataloader:
        n += i["input_ids"].shape[0]

    assert n == num_records * 2


@pytest.mark.usefixtures("_convert_rdata_to_litdata")
def test_streaming_train(
    streaming_panglao_parameters, gene2vec_fields: list[config.FieldInfo]
):
    datamodule = StreamingPanglaoDBDataModule(**streaming_panglao_parameters)
    datamodule.prepare_data()
    datamodule.setup()
    datamodule.max_length = 16
    trainer_config = config.TrainerConfig(
        losses=default_mlm_losses_from_fields(gene2vec_fields)
    )
    model_config = config.SCBertConfig(
        batch_size=datamodule.batch_size,
        fields=gene2vec_fields,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_size=16,
        max_position_embeddings=datamodule.max_length,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_config = config.TrainingTaskConfig(
            default_root_dir=tmpdir,
            max_epochs=1,
            max_steps=3,
            val_check_interval=3,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[],
        )
        training_module = MLMTrainingModule(
            model_config, trainer_config, tokenizer=datamodule.tokenizer
        )
        pl_trainer = make_trainer_for_task(task_config)
        train(
            pl_trainer,
            pl_data_module=datamodule,
            pl_module=training_module,
            task_config=task_config,
        )
