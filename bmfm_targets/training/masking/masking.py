import logging
from collections.abc import Mapping

import torch

from bmfm_targets.config import FieldInfo
from bmfm_targets.training.masking.strategy import MaskingStrategy

logger = logging.getLogger(__name__)


class Masker:
    """
    Masks inputs for masked-language model training.

    This is an internal class that is used by the collator and data module.
    Options are documented in the data module class, where they are exposed to the user.
    """

    def __init__(
        self,
        change_ratio: float,
        mask_ratio: float,
        switch_ratio: float,
        tokenizer,
        comask_across_fields: bool = False,
        prevent_attention_to_masked: bool = False,
        masking_strategy: MaskingStrategy | None = None,
    ):
        self.change_ratio = change_ratio
        self.mask_ratio = mask_ratio
        self.switch_ratio = switch_ratio
        self.tokenizer = tokenizer
        self.comask_across_fields = comask_across_fields
        self.prevent_attention_to_masked = prevent_attention_to_masked
        self.masking_strategy = masking_strategy
        self.__post_init__()

    def __post_init__(self):
        if self.change_ratio < 0 or self.change_ratio > 1:
            raise ValueError(
                "Change ratio should be between 0 and 1. 0 means no change and 1 means all tokens are changed"
            )
        if self.mask_ratio < 0 or self.mask_ratio > 1:
            raise ValueError("Mask ratio should be between 0 and 1")

        if self.switch_ratio < 0 or self.switch_ratio > 1:
            raise ValueError("Switch ratio should be between 0 and 1")

        if self.mask_ratio + self.switch_ratio > 1:
            raise ValueError(
                "Mask ratio + Switch ratio should be less than or equal to 1"
            )
        if self.switch_ratio > 0 and self.comask_across_fields:
            raise ValueError("switch ratio>0  does not work with comasking")

    def mask_inputs(
        self,
        fields: list[FieldInfo],
        batch: dict[str, Mapping[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """

        Mask inputs from all `mask_field_names`s for language modeling.


        Args:
        ----
            fields (list[FieldInfo]): all fields for the model
            batch (dict[str, Mapping[str, torch.Tensor]]): batch of fields and input_ids


        Returns:
        -------
            tuple[torch.Tensor, dict[str, torch.Tensor]]: input_ids and labels for all fields

        """
        input_fields = [f for f in fields if f.is_input]
        masked_fields = [f for f in input_fields if f.is_masked]
        labels = {}

        random_tensor = torch.rand_like(
            batch[input_fields[0].field_name]["input_ids"],
            layout=torch.strided,
            dtype=torch.float,
            device=batch[input_fields[0].field_name]["input_ids"].device,
        )

        for field in masked_fields:
            mask_probs = self.get_mask_probs(batch, masked_fields)
            if not self.comask_across_fields:
                random_tensor = torch.rand_like(
                    batch[field.field_name]["input_ids"],
                    layout=torch.strided,
                    dtype=torch.float,
                    device=batch[field.field_name]["input_ids"].device,
                )
            _, field_labels = self.mask_single_field(
                field,
                batch[field.field_name],
                random_tensor,
                mask_probs,
                batch.get(f"label_{field.field_name}", None),
            )
            labels[field.field_name] = field_labels
            logger.debug(
                f"uncorrected mask rate: {1 - (field_labels == -100).count_nonzero() / field_labels.flatten().shape[0]}"
            )

        field_input_ids = [
            batch[field.field_name]["input_ids"] for field in input_fields
        ]

        attention_mask: torch.Tensor = batch[input_fields[0].field_name][
            "attention_mask"
        ]
        if self.prevent_attention_to_masked:
            if not self.comask_across_fields and len(masked_fields) > 1:
                raise ValueError(
                    "prevent_attention_to_masked is not defined when masking varies between fields"
                )

            attention_mask = prevent_attention_to_masked(
                batch,
                masked_fields[0],
                attention_mask,
                mask_id=self.tokenizer.mask_token_id,
            )
        input_ids = torch.stack(field_input_ids, dim=1)
        return input_ids, labels, attention_mask

    def mask_single_field(
        self,
        field: FieldInfo,
        field_encoding: Mapping[str, torch.Tensor],
        random_tensor: torch.Tensor,
        mask_probs: torch.Tensor | None = None,
        label_encoding: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mask tensor for a single field with non-uniform masking probabilities.

        Args:
        ----
            field (FieldInfo): field config object
            field_encoding (Mapping[str, torch.Tensor]): encoding for field
            mask_probs (torch.Tensor | None): probabilities for masking specific tokens
            label_encoding (Mapping[str, torch.Tensor] | None, optional):
            labels other than input to use. Defaults to None, in which case the labels are
            the inputs (normal MLM behavior).

        Returns:
        -------
            tuple[torch.Tensor, torch.Tensor]: input_ids and label_ids after masking

        """
        input_ids = field_encoding["input_ids"]
        if label_encoding is not None:
            label_ids = label_encoding["input_ids"].clone().detach()
        else:
            label_ids = input_ids.clone().detach()
        special_tokens_mask = field_encoding["special_tokens_mask"]

        if mask_probs is None:
            mask_probs = torch.full_like(random_tensor, fill_value=self.change_ratio)
        else:
            # Normalize probabilities so that on average the right number are changed
            mask_probs *= self.change_ratio / mask_probs.mean()
            mask_probs = torch.clamp(mask_probs, 0, 1)

        mask_probs[special_tokens_mask.bool()] = 0

        # Keep indices based on probabilities
        keep_indices = random_tensor.gt(mask_probs)
        label_ids[keep_indices] = -100

        # Masking tokens which are not preserved
        mask_indices = random_tensor.le(mask_probs * self.mask_ratio)

        if field.tokenization_strategy == "continuous_value_encoder":
            mask_value = -(self.tokenizer.mask_token_id + 1)
        else:
            mask_value = self.tokenizer.mask_token_id

        input_ids[mask_indices] = mask_value

        # Switching tokens which are not preserved and not masked

        if self.switch_ratio > 0:
            random_tokens = _draw_random_tokens(
                field.field_name, self.tokenizer, label_ids
            )
            mask_or_switch_probs = mask_probs * (self.mask_ratio + self.switch_ratio)
            switch_indices = random_tensor.le(mask_or_switch_probs) & (~mask_indices)
            input_ids[switch_indices] = random_tokens[switch_indices]
        return input_ids, label_ids

    def get_mask_probs(
        self, batch: dict[str, Mapping[str, torch.Tensor]], fields: list[FieldInfo]
    ) -> torch.Tensor:
        if self.masking_strategy is not None:
            mask_probs = self.masking_strategy.get_mask_probs(batch)
        else:
            mask_probs = torch.ones_like(
                batch[fields[0].field_name]["input_ids"]
            ).float()
        if not self.comask_across_fields:
            already_masked = self.find_masked_or_unk(batch, fields)
            mask_probs[already_masked] = 0

        return mask_probs

    def find_masked_or_unk(
        self,
        batch: dict[str, Mapping[str, torch.Tensor]],
        other_mask_fields: list[FieldInfo],
    ):
        ineligible_list = []
        for field in other_mask_fields:
            mask_id = self.tokenizer.mask_token_id
            unk_id = self.tokenizer.unk_token_id
            if field.tokenization_strategy == "continuous_value_encoder":
                mask_id = -(mask_id + 1)
            already_masked = batch[field.field_name]["input_ids"] == mask_id
            if field.tokenization_strategy != "continuous_value_encoder":
                # with continuous values there are no unk and preventing unk_id=0
                # means never masking 0
                is_unk = batch[field.field_name]["input_ids"] == unk_id
                ineligible_list.append(already_masked | is_unk)
            else:
                ineligible_list.append(already_masked)
        already_masked = torch.stack(ineligible_list, dim=-1)
        return already_masked.any(dim=-1)

    def update_token_masking_probs(self, token_probs: dict[str, float]):
        return self.masking_strategy.update_token_masking_probs(token_probs)


def _draw_random_tokens(field_name, tokenizer, label_tensor):
    min_val = len(tokenizer.all_special_ids)
    max_val = tokenizer.field_vocab_size(field_name)
    return torch.randint_like(label_tensor, low=min_val, high=max_val)


def prevent_attention_to_masked(
    batch: dict[str, dict[str, torch.Tensor]],
    masked_field: FieldInfo,
    attention_mask: torch.Tensor,
    mask_id: int,
) -> torch.Tensor:
    """
    Constructs an attention mask for a batch, ensuring masked tokens only attend to unmasked tokens
    and themselves, while padding and special tokens are excluded from attention.

    Args:
    ----
        batch: Dictionary containing tokenized input fields.
        masked_field: Object containing field metadata, including tokenization strategy and field name.
        attention_mask: Binary tensor indicating which tokens should be attended to.
        mask_id: Token ID representing masked tokens.

    Returns:
    -------
        A boolean tensor representing the final attention mask.
    """
    input_ids = batch[masked_field.field_name]["input_ids"]

    if masked_field.tokenization_strategy == "continuous_value_encoder":
        mask_id = -(mask_id + 1)

    is_masked = (input_ids == mask_id).long()
    is_unmasked = attention_mask & ~is_masked

    unmasked_attention = is_unmasked.unsqueeze(1) & is_unmasked.unsqueeze(2)
    masked_to_unmasked_attention = (
        is_unmasked.unsqueeze(1)
        & is_masked.unsqueeze(2)
        & ~batch[masked_field.field_name]["special_tokens_mask"].unsqueeze(1)
    )

    batch_size, seq_len = input_ids.shape
    masked_self_attention = torch.zeros(
        batch_size, seq_len, seq_len, device=input_ids.device, dtype=torch.long
    )
    masked_self_attention[:, torch.arange(seq_len), torch.arange(seq_len)] = is_masked

    combined_attention = (
        unmasked_attention | masked_to_unmasked_attention | masked_self_attention
    )
    final_attention = (
        combined_attention & attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
    )

    return final_attention
