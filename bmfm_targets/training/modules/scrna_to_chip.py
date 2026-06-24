import logging

import torch

from bmfm_targets.training.losses.ot.sinkhorn import sinkhorn_divergence
from bmfm_targets.training.losses.utils import calculate_losses

from .sequence_labeling import SequenceLabelingTrainingModule

logger = logging.getLogger(__name__)


class ScrnaToChipTranslationModule(SequenceLabelingTrainingModule):
    """scRNA → ChIP population-OT translation module.

    Runs standard WCED reconstruction losses and adds a Sinkhorn-divergence
    population-OT term comparing the predicted ChIP population [B, n_genes]
    against the true ChIP population [M, n_genes] supplied in
    batch["chip_population"] (injected by scRNA2ChIPDataModule when
    use_ot_batching=True).

    ot_weight and wced_weight are passed as extra_kwargs in the YAML trainer
    section and forwarded here via **kwargs.
    """

    def __init__(
        self,
        model_config,
        trainer_config,
        ot_weight: float = 1.0,
        wced_weight: float = 1.0,
        ot_eps: float = 1.0,
        ot_n_iters: int = 100,
        cost: str = "euclidean",
        **kwargs,
    ):
        super().__init__(model_config, trainer_config, **kwargs)
        self.ot_weight = ot_weight
        self.wced_weight = wced_weight
        self.ot_eps = ot_eps
        self.ot_n_iters = ot_n_iters
        self.cost = cost

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", {})

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        all_losses = calculate_losses(self.loss_tasks, outputs.logits, labels)
        all_losses["loss"] = self.wced_weight * all_losses["loss"]

        chip_population = batch.get("chip_population")  # [M, n_genes]
        if chip_population is not None:
            # WCED logits: [B, 1, vocab_size, n_outputs] — decode token is dim 1
            pred = outputs.logits.get("label_expressions_wced")  # [B, seq_len, vocab_size]
            if pred is not None:
                pred = pred[:, 0, :]  # [B, vocab_size] — CLS decode token
                B, M = pred.shape[0], chip_population.shape[0]
                if B >= 2 and M >= 2:
                    ot_loss = sinkhorn_divergence(
                        pred,
                        chip_population.float(),
                        eps=self.ot_eps,
                        n_iters=self.ot_n_iters,
                        cost=self.cost,
                    )
                    all_losses["loss"] = all_losses["loss"] + self.ot_weight * ot_loss
                    all_losses["ot_loss"] = ot_loss
                else:
                    logger.warning(f"Skipping OT loss: B={B}, M={M} (need ≥2 each)")

        self.log_losses(all_losses, input_ids.shape[0], split="train")
        loss = all_losses["loss"]
        return loss if loss != 0.0 else None
