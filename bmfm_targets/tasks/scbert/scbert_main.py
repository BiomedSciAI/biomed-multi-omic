import logging
import os
from copy import deepcopy

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from bmfm_targets import config
from bmfm_targets.tasks.task_utils import (
    main_config,
    main_run,
    start_clearml_logger,
    update_task_from_trainer,
)

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="base_scbert_config", node=config.SCBertMainHydraConfigSchema)


@hydra.main(config_path=".", config_name="scbert_train", version_base="1.2")
def main(cfg: config.SCBertMainHydraConfigSchema) -> None:
    os.umask(0)
    if not isinstance(cfg.task, list | dict | ListConfig | DictConfig):
        raise ValueError("task must be a list[dict], dict, ListConfig, or DictConfig")

    if hasattr(cfg, "track_clearml") and cfg.track_clearml:
        clearml_logger = start_clearml_logger(**cfg.track_clearml)
    else:
        clearml_logger = None

    tasks = [cfg.task] if not isinstance(cfg.task, ListConfig) else cfg.task
    trainer = None
    for task in tasks:
        if trainer is not None:
            task = update_task_from_trainer(task, trainer)
        this_cfg = deepcopy(cfg)
        this_cfg.task = task
        cfg_obj = main_config(this_cfg)

        trainer = main_run(
            cfg_obj.task,
            cfg_obj.model,
            cfg_obj.data_module,
            cfg_obj.trainer,
            clearml_logger,
        )


if __name__ == "__main__":
    main()
