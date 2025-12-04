import logging
import re

import torch

from bmfm_targets.training.sample_transforms import transform_inputs


def select_input_labels(
    label_tensor: torch.Tensor,
    sample_idx: int,
    lookup_ids: torch.Tensor,
    lookup_vals: torch.Tensor,
    input_ids: torch.Tensor,
) -> None:
    """Assign labels for genes present in input."""
    input_mask = torch.isin(lookup_ids, input_ids[sample_idx, 0])
    label_tensor[sample_idx, lookup_ids[input_mask]] = lookup_vals[input_mask]


def select_non_input_labels(
    label_tensor: torch.Tensor,
    sample_idx: int,
    lookup_ids: torch.Tensor,
    lookup_vals: torch.Tensor,
    input_ids: torch.Tensor,
) -> None:
    """Assign labels for genes NOT present in input."""
    non_input_mask = ~torch.isin(lookup_ids, input_ids[sample_idx, 0])
    label_tensor[sample_idx, lookup_ids[non_input_mask]] = lookup_vals[non_input_mask]


def select_all_labels(
    label_tensor: torch.Tensor,
    sample_idx: int,
    lookup_ids: torch.Tensor,
    lookup_vals: torch.Tensor,
    input_ids: torch.Tensor,
) -> None:
    """Assign labels for all genes."""
    label_tensor[sample_idx, lookup_ids] = lookup_vals


LABEL_SET_FUNCTIONS = {
    "input": select_input_labels,
    "non_input": select_non_input_labels,
    "all": select_all_labels,
}


class MaskingStrategy:
    def __init__(
        self,
        tokenizer,
        lookup_field_name="genes",
        pattern_weights: list[tuple[str, float]] | None = None,
        use_for_validation: bool = False,
    ):
        """
        Base masking strategy that applies regex-based weighting to token masking probabilities.

        Args:
        ----
            tokenizer (MultiFieldTokenizer): Tokenizer handling multiple fields.
            lookup_field_name (str, optional): Field name used to identify tokens
              for masking. Defaults to "genes".
            pattern_weights (list[tuple[str, float]] | None, optional): List of tuples
              containing regex patterns and their corresponding weights
              (neutral weight is 1). Defaults to None.
            use_for_validation (bool, optional): If False, validation uses uniform
              masking probabilities. Defaults to False.
        """
        self.tokenizer = tokenizer
        self.lookup_field_name = lookup_field_name
        fv = tokenizer.get_field_vocab(self.lookup_field_name)
        self.token_masking_probs = torch.ones(size=(len(fv),))
        self.token_masking_probs.share_memory_()
        self.pattern_weights = pattern_weights
        self.use_for_validation = use_for_validation
        if self.pattern_weights:
            self.update_token_probs_from_pattern_weights()

    def get_mask_probs(self, batch: dict) -> torch.Tensor:
        """
        Retrieve per-token masking probabilities for a given batch.

        Args:
        ----
            batch (dict): Batch dictionary containing tokenized input data. Must have
                `batch[self.lookup_field_name]["input_ids"]`.

        Returns:
        -------
            torch.Tensor: Tensor of masking probabilities corresponding to the input tokens.
        """
        return self.token_masking_probs[batch[self.lookup_field_name]["input_ids"]]

    def update_token_probs_from_pattern_weights(self):
        """
        Update token masking probabilities based on the specified regex pattern weights.

        This method scans the vocabulary for tokens matching given regex patterns and
        scales their masking probabilities multiplicatively according to the specified weights.
        """
        assert self.pattern_weights is not None
        vocab = self.tokenizer.get_field_vocab(self.lookup_field_name)
        for regex, weight in self.pattern_weights:
            matching_ids = self.find_ids_matching_pattern(vocab, regex)
            self.token_masking_probs[matching_ids] *= weight

    @staticmethod
    def find_ids_matching_pattern(vocab, regex):
        """
        Find token IDs in the vocabulary that match a given regex pattern.

        Args:
        ----
            vocab (dict): Dictionary mapping token strings to their corresponding IDs.
            regex (str): Regular expression pattern to match against tokens.

        Returns:
        -------
            list[int]: List of token IDs that match the pattern.
        """
        pattern = re.compile(regex)
        matching_ids = [tok_id for tok, tok_id in vocab.items() if pattern.match(tok)]
        return matching_ids


class WCEDMasker:
    """
    Whole Cell Expression Decoder Masker.

    Prepares labels for predicting gene expression values, distinguishing between
    genes in the input sequence vs. non_input genes. Creates "input" and "non_input"
    label sets for training models on complete cellular expression profiles.
    """

    def __init__(
        self,
        tokenizer,
        label_sets: list[str] | None = None,
        lookup_field_name: str = "genes",
        value_field_name: str = "expressions",
        sample_transforms_kwargs: dict | None = None,
        **kwargs,
    ):
        """Initialize WCED masker with tokenizer and optional label sets."""
        self.tokenizer = tokenizer
        self.lookup_field_name = lookup_field_name
        self.value_field_name = value_field_name
        self.label_sets = label_sets or ["non_input", "input", "all"]
        self.sample_transforms_kwargs = sample_transforms_kwargs or {}
        self.lookup_field_vocab_len = len(
            self.tokenizer.get_field_vocab(self.lookup_field_name)
        )
        self.lookup_field_tokenizer = self.tokenizer.get_field_tokenizer(
            self.lookup_field_name
        )

    @property
    def _sample_transforms_transform_inputs_kwargs(self) -> dict:
        """
        Get transformation parameters with defaults.

        These are used for transforming the samples to be used for labels.
        So limitations on length, ordering operations, and noising augmentations
        should all be off.
        """
        kwargs = {
            "label_columns": [],
            "sequence_order": None,
            "log_normalize_transform": None,
            "median_normalization": None,
            "rda_transform": None,
            "pad_zero_expression_strategy": None,
            "max_length": None,
            "sequence_dropout_factor": None,
            "map_orthologs": None,
            "renoise": None,
            "selective_dropout_weights": None,
            "limit_input_tokens": None,
        }
        kwargs.update(self.sample_transforms_kwargs)
        warn_if_unset_keys = ["log_normalize_transform", "median_normalization"]
        for key in warn_if_unset_keys:
            if kwargs[key] is None:
                logging.warning(
                    f"{key} is not set for WCED masker, defaulting to False"
                )
                kwargs[key] = False
        return kwargs

    def mask_inputs(
        self, fields: list, batch: dict
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]], torch.Tensor]:
        """
        Process batch to create input tensors and expression labels.

        Returns: (input_ids, labels, attention_mask)
        """
        input_fields = [f for f in fields if f.is_input]
        attention_mask = batch[input_fields[0].field_name]["attention_mask"]
        input_ids = torch.stack(
            [batch[field.field_name]["input_ids"] for field in input_fields], dim=1
        )

        batch_size = input_ids.shape[0]

        # Initialize label tensors
        label_tensor_dict = {
            label_set: -100
            * torch.ones(batch_size, self.lookup_field_vocab_len, dtype=torch.float)
            for label_set in self.label_sets
        }

        # Transform examples and populate labels
        examples = transform_inputs(
            batch["mfi"], fields, **self._sample_transforms_transform_inputs_kwargs
        )
        tokenized_lookup_field = self.lookup_field_tokenizer(
            [mfi[self.lookup_field_name] for mfi in examples],
            is_split_into_words=True,
            add_special_tokens=False,
            return_attention_mask=True,
            truncation=False,
            padding="longest",
            return_token_type_ids=False,
            return_tensors="pt",
        )

        for sample_idx, mfi in enumerate(examples):
            lookup_vals = torch.tensor(mfi[self.value_field_name], dtype=torch.float)
            lookup_ids = tokenized_lookup_field["input_ids"][sample_idx][
                tokenized_lookup_field["attention_mask"][sample_idx].bool()
            ]
            for label_set in self.label_sets:
                LABEL_SET_FUNCTIONS[label_set](
                    label_tensor_dict[label_set],
                    sample_idx,
                    lookup_ids,
                    lookup_vals,
                    input_ids,
                )

        return input_ids, {self.value_field_name: label_tensor_dict}, attention_mask
