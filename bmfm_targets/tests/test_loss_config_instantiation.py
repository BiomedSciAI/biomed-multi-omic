"""
Tests for loss task configuration and instantiation.

This module tests the new architecture where loss tasks are instantiated
at the config layer and bound at the training module layer.
"""

import pytest

from bmfm_targets.config import FieldInfo, LabelColumnInfo, TrainerConfig
from bmfm_targets.training.losses import (
    CrossEntropyObjective,
    FieldSource,
    LabelSource,
    LossTask,
    MSEObjective,
    loss_dict_to_task,
)


def test_old_style_dict_to_loss_task():
    """Test conversion of old-style dict config to LossTask."""
    fields = [FieldInfo(field_name="expressions", vocab_size=100)]
    label_columns = [LabelColumnInfo(label_column_name="cell_type", n_unique_values=10)]

    # Old-style dict for label loss
    loss_config = {
        "label_column_name": "cell_type",
        "name": "cross_entropy",
        "weight": 1.0,
    }
    loss_task = loss_dict_to_task(loss_config, fields, label_columns)

    assert isinstance(loss_task, LossTask)
    assert isinstance(loss_task.source, LabelSource)
    assert isinstance(loss_task.objective, CrossEntropyObjective)
    assert loss_task.weight == 1.0


def test_old_style_dict_field_loss():
    """Test conversion of old-style dict config for field loss."""
    fields = [FieldInfo(field_name="expressions", vocab_size=100)]
    label_columns = []

    # Old-style dict for field loss
    loss_config = {"field_name": "expressions", "name": "mse", "weight": 1.0}
    loss_task = loss_dict_to_task(loss_config, fields, label_columns)

    assert isinstance(loss_task, LossTask)
    assert isinstance(loss_task.source, FieldSource)
    assert isinstance(loss_task.objective, MSEObjective)
    assert loss_task.weight == 1.0


def test_hydra_config_error_in_loss_dict_to_task():
    """Test that Hydra configs raise error in loss_dict_to_task."""
    fields = [FieldInfo(field_name="expressions", vocab_size=100)]
    label_columns = []

    # Hydra-style config (should not be passed to loss_dict_to_task)
    loss_config = {
        "_target_": "bmfm_targets.training.losses.LossTask",
        "source": {"_target_": "bmfm_targets.training.losses.FieldSource"},
    }

    # This should work in loss_dict_to_task since it doesn't check for _target_
    # The error should come from get_loss_tasks instead
    # Actually, loss_dict_to_task will fail because it looks for field_name or label_column_name
    with pytest.raises(
        ValueError, match="must have 'field_name' or 'label_column_name'"
    ):
        loss_dict_to_task(loss_config, fields, label_columns)


def test_trainer_config_with_dict_losses():
    """Test that TrainerConfig accepts dict-style losses (backward compat)."""
    # This is how tests currently create configs
    trainer_config = TrainerConfig(
        losses=[
            {"label_column_name": "cell_type", "name": "cross_entropy"},
            {"field_name": "expressions", "name": "mse"},
        ]
    )

    assert len(trainer_config.losses) == 2
    assert isinstance(trainer_config.losses[0], dict)
    assert isinstance(trainer_config.losses[1], dict)


def test_trainer_config_with_loss_task_objects():
    """Test that TrainerConfig accepts LossTask objects (new style)."""
    loss_task1 = LossTask(
        source=LabelSource("cell_type"),
        objective=CrossEntropyObjective(),
        weight=1.0,
    )
    loss_task2 = LossTask(
        source=FieldSource("expressions"),
        objective=MSEObjective(),
        weight=1.0,
    )

    trainer_config = TrainerConfig(losses=[loss_task1, loss_task2])

    assert len(trainer_config.losses) == 2
    assert isinstance(trainer_config.losses[0], LossTask)
    assert isinstance(trainer_config.losses[1], LossTask)


def test_mixed_config_styles():
    """Test that TrainerConfig accepts mixed dict and LossTask objects."""
    loss_task = LossTask(
        source=LabelSource("cell_type"),
        objective=CrossEntropyObjective(),
        weight=1.0,
    )

    trainer_config = TrainerConfig(
        losses=[
            loss_task,
            {"field_name": "expressions", "name": "mse"},
        ]
    )

    assert len(trainer_config.losses) == 2
    assert isinstance(trainer_config.losses[0], LossTask)
    assert isinstance(trainer_config.losses[1], dict)


def test_loss_task_properties():
    """Test that LossTask has all expected properties."""
    loss_task = LossTask(
        source=LabelSource("cell_type"),
        objective=CrossEntropyObjective(),
        weight=1.0,
        loss_group="test_group",
    )

    # Bind to set up properties
    label_columns = [LabelColumnInfo(label_column_name="cell_type", n_unique_values=10)]
    loss_task.bind([], label_columns, None)

    # Test properties
    assert loss_task.name == "cell_type_test_group_cross_entropy"
    assert loss_task.loss_display_name == "cell_type_test_group_cross_entropy_loss"
    assert loss_task.metric_key == "cell_type_test_group"
    assert loss_task.source.name == "cell_type"
    # LabelSource gets decoder_key with suffix from objective
    assert loss_task.decoder_key == "cell_type_token_scores"
    assert loss_task.logit_key == "cell_type_token_scores"  # Alias for decoder_key


def test_regression_objective_output_size():
    """Test that regression objectives return output_size=1."""
    mse_obj = MSEObjective()
    assert mse_obj.output_size == 1

    from bmfm_targets.training.losses import MAEObjective

    mae_obj = MAEObjective()
    assert mae_obj.output_size == 1


# Made with Bob
