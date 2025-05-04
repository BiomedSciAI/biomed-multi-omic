import re

import torch


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
