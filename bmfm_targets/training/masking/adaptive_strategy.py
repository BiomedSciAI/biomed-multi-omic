import pandas as pd
import pytorch_lightning as pl
import torch

from bmfm_targets.tokenization import MultiFieldTokenizer
from bmfm_targets.training.callbacks import logger
from bmfm_targets.training.masking.strategy import MaskingStrategy, WCEDMasker


class TokenErrorUpdateCallback(pl.Callback):
    """
    Callback for updating token errors.

    Added automatically when DataModule initiated with `TokenProbabilityMaskingStrategy`.
    Requires the token level errors to be calculated and saved to the trainer's
    `token_level_errors` attribute.
    """

    def __init__(
        self,
        error_column_name="gene_err",
        token_level_error_key=None,
        n_bins=100,
        error_masking_relation: str = "increasing",
        use_for_validation: bool = False,
    ) -> None:
        self.error_column_name = error_column_name
        self.n_bins = n_bins
        self.token_level_error_key = token_level_error_key
        self.error_masking_relation = error_masking_relation
        self.use_for_validation = use_for_validation
        super().__init__()

    def on_validation_end(self, trainer, pl_module):
        # Get token errors from the LightningModule
        if len(pl_module.token_level_errors) == 0:
            logger.warning(
                "No gene level errors available to update masking. "
                "No adaptive masking will take place."
            )
            return
        if len(pl_module.token_level_errors) == 1:
            key, errors = [*pl_module.token_level_errors.items()][0]
            logger.info(f"Using token_level_errors['{key}']")
        elif self.token_level_error_key is not None:
            errors = pl_module.token_level_errors[self.token_level_error_key]
        else:
            logger.warning(
                "Multiple gene level errors available and none selected. "
                "No adaptive masking will take place. "
                f"Options: {pl_module.token_level_errors.keys()}"
            )
            return

        if errors is not None:
            # Compute token probabilities
            token_probs = self.calculate_token_probs(errors)
            self.update_masker_token_probs(trainer, token_probs)
            # Update masking strategy in the DataModule

    def update_masker_token_probs(
        self, trainer: pl.Trainer, token_probs: dict[str, float]
    ):
        """
        Load the masking_strategy object from the datamodule or dataloader.

        Depending on how Trainer.fit() is called, there will be either a datamodule
        or dataloaders. The masking_strategy object is shared between them, but where
        it is stored needs to be deduced.

        Args:
        ----
            trainer (pl.Trainer): the lightning trainer

        Raises:
        ------
            ValueError: if there are no valid dataloaders at all. This would only happen
              if this function is called outside the fit/test loop.

        Returns:
        -------
            MaskingStrategy | None: the masking strategy or None if there is no masking
              strategy defined

        """
        import os

        logger.info(
            f"on_validation_end on rank {trainer.global_rank} | pid {os.getpid()}"
        )

        if getattr(trainer, "train_dataloader", None):
            collator = trainer.train_dataloader.collate_fn
            self._update_dataloader_token_probs(token_probs, collator)
        if self.use_for_validation:
            if getattr(trainer, "val_dataloaders", None):
                collator = trainer.val_dataloaders.collate_fn
                self._update_dataloader_token_probs(token_probs, collator)
            if getattr(trainer, "test_dataloaders", None):
                collator = trainer.test_dataloaders.collate_fn
                self._update_dataloader_token_probs(token_probs, collator)

    def _update_dataloader_token_probs(self, token_probs, collator):
        masker = (
            collator.masker
            if hasattr(collator, "masker")
            else collator.collate_fn.masker
        )
        masker.update_token_masking_probs(token_probs)
        if torch.distributed.is_initialized():
            logger.info("broadcasting token_masking_probs tensor")
            torch.distributed.broadcast(
                masker.masking_strategy.token_masking_probs, src=0
            )

    def calculate_token_probs(self, errors: pd.DataFrame) -> dict[str, float]:
        """
        Calculate token masking probabilities based on token error dataframe.

        This makes use of the `error_column_name` attribute to choose which error
        definition to use to calculate masking probabilities. It transforms the errors
        using a quantile transform and shifts the values from 1/n_bins to 1 so that
        nothing has zero probability.

        Args:
        ----
          errors (pd.DataFrame): the token_level error dataframe as produced, eg, by
            `get_gene_level_expression_error`.

        Returns:
        -------
            dict[str,float]: tokens and masking probabilities. The probabilities do not
              need to be valid probabilities, they will be rescaled by the masking
              function.

        """
        error_to_use = errors[self.error_column_name]
        if self.error_masking_relation == "decreasing":
            error_to_use = error_to_use.max() - error_to_use
        token_probs = pd.cut(error_to_use, bins=self.n_bins, labels=False)
        # we don't want any zeros and we want normalized to 1
        token_probs = (token_probs + 1) / (self.n_bins + 1)

        return token_probs.to_dict()


class AdaptiveMaskingStrategy(MaskingStrategy):
    def __init__(
        self,
        tokenizer: MultiFieldTokenizer,
        lookup_field_name="genes",
        pattern_weights: list[tuple[str, float]] | None = None,
        temperature: float | None = None,
        callback: pl.Callback | None = None,
        use_for_validation: bool = True,
    ):
        super().__init__(
            tokenizer=tokenizer,
            lookup_field_name=lookup_field_name,
            pattern_weights=pattern_weights,
            use_for_validation=use_for_validation,
        )
        self.temperature = temperature
        if callback is None:
            callback = TokenErrorUpdateCallback(use_for_validation=use_for_validation)
        self.callback = callback

    def get_trainer_callback(self):
        return self.callback

    def update_token_masking_probs(self, token_probs: dict[str, float]):
        ft = self.tokenizer.get_field_tokenizer(self.lookup_field_name)
        ft.backend_tokenizer.pre_tokenizer = None
        assert ft.backend_tokenizer.pre_tokenizer is None
        token_ids = ft(
            [*token_probs.keys()],
            is_split_into_words=True,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )["input_ids"].squeeze()
        token_prob_vec = torch.tensor([*token_probs.values()])
        if self.temperature is not None:
            token_prob_vec = torch.functional.F.softmax(
                token_prob_vec / self.temperature, dim=0
            )
        self.token_masking_probs[token_ids] = token_prob_vec


class AdaptiveWCEDMasker(AdaptiveMaskingStrategy, WCEDMasker):
    def __init__(self, tokenizer, *args, **kwargs):
        AdaptiveMaskingStrategy.__init__(
            self,
            tokenizer=tokenizer,
            *args,
            **kwargs,
            callback=TokenErrorUpdateCallback(
                token_level_error_key="expressions_non_input_genes",
                error_masking_relation="decreasing",
            ),
        )
        WCEDMasker.__init__(self, tokenizer=tokenizer, *args, **kwargs)
        tokens = [*self.tokenizer.get_field_vocab(self.lookup_field_name)]
        self.selective_dropout_weights = pd.Series(
            index=tokens, data=[1.0] * len(tokens)
        )

    def update_token_masking_probs(self, token_probs: dict[str, float]):
        new_probs = pd.Series(token_probs)
        if self.temperature is not None:
            new_probs[:] = torch.functional.F.softmax(
                new_probs.values / self.temperature, dim=0
            ).numpy()
        self.selective_dropout_weights.loc[new_probs.index] = new_probs
