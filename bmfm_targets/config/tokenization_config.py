import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class TokenizerConfig:
    """
    Tokenizer configuration.

    Attrs:
        identifier (str) : identifier for tokenizer. Either a simple string naming a
            packaged tokenizer ('gene2vec' or 'all_genes') or a path to a directory
            containing files required to instantiate MultifieldTokenizer.
    """

    identifier: str = "all_genes"


@dataclass
class PreTrainedEmbeddingConfig:
    """
    Pre trained embeddings configuration.
    A file with pre computed embeddings. This file is a .txt file (gene2vec format) with space as the deliminator. The first row is the size of the embedding matrix.

    Attrs:
        filename (str) : Path to the pre computed embeddings file.
        embedding_indices_to_freeze (list) : List with indices of tokens that exist in the embedding file. These embeddings will be frozen during training by
        setting their gradient to zero. Tokens that do not exist in the pre computed embedding file will be initiated as random and update during training.
        pre_trained_indices_to_use (list): List of indices to use from the pre computed embeddings file.
    """

    filename: str | None = None
    embedding_indices_to_freeze: list[int] | None = None
    pre_trained_indices_to_use: list[int] | None = None

    def get_indices_of_pretrained_token_embeddings(
        self, field_name, tokenizer
    ) -> tuple[list[int], list[int]]:
        """
        Get the indices to freeze in the embedding matrix, and the indices of the pre-computed embedding to use.
        Attrs:
            field_name (str): The field name
            tokenizer (Tokenizer): The model tokenizer.
        """
        pretrained_embeddings = self.load_pretrained_embeddings()

        indices_of_pre_computed_in_tokenizer = self.get_indices_of_tokens(
            field_name, pretrained_embeddings.index.values, tokenizer
        )
        (
            embedding_indices_to_freeze,
            pre_trained_indices_to_use,
        ) = self.identify_indices_to_keep(
            indices_of_pre_computed_in_tokenizer, tokenizer, field_name
        )
        return embedding_indices_to_freeze, pre_trained_indices_to_use

    def load_pretrained_embeddings(self):
        """
        Load pretrained embeddings from .txt file.
        The format of the file: a .txt file with space separators and the first row
        has the size of the embeddings. for example for five embeddings with the length of 5 the first row will be: 5 5.

        """
        return pd.read_csv(
            self.filename,
            sep=" ",
            index_col=0,
            header=None,
            skiprows=1,
        )

    def get_indices_of_tokens(self, field_name, pretrained_tokens, tokenizer):
        """
        Get the indices of tokens from a pre computed embedding file from the tokenizer.

        Attrs:
            field_name (str): The field name
            pretrained_tokens (array): The array of tokens in the pre computed file
            tokenizer (Tokenizer): The model tokenizer
        """
        idx_of_tokens = tokenizer.convert_field_tokens_to_ids(
            field_name, tokens=pretrained_tokens
        )
        return np.array(idx_of_tokens)

    def identify_indices_to_keep(
        self, indices_of_pre_computed_in_tokenizer, tokenizer, field_name
    ):
        """
        Identify what indices in the pre computed embeddings matrix to use and update the indices to freeze to not include the unk char.

        Attrs:
            indices_of_pre_computed_in_tokenizer (array): List with the index of the tokens from pre computed embeddings matrix in tokenizer
            tokenizer (Tokenizer): The model tokenizer
            field_name (str): The field name
        """
        unk_id = tokenizer.convert_field_tokens_to_ids(
            field_name, [tokenizer.unk_token]
        )
        mask = indices_of_pre_computed_in_tokenizer != unk_id
        warnings.warn(
            f"There a total of {sum(mask)} pre computed embeddings that are in the {field_name} tokenizer and will be used"
        )
        pre_trained_indices_to_use = np.nonzero(mask)[0].tolist()
        embedding_indices_to_freeze = indices_of_pre_computed_in_tokenizer[
            mask
        ].tolist()
        return embedding_indices_to_freeze, pre_trained_indices_to_use


@dataclass(eq=True, repr=True)
class FieldInfo:
    """
    Represents information about a field.

    Args:
    ----
        field_name (str): the name of the field
        vocab_size (int): The size of the vocabulary. This depends on the tokenizer and
          may not be known at instantiation time.
        vocab_update_strategy (str): The strategy to use for updating the vocabulary. Defaults to "static".
        pretrained_embedding (PreTrainedEmbeddingConfig): The pre trained embedding object, with the path to the embeddings.
        is_masked (bool): Whether to mask the field
        decode_modes (list[str]): Output modes to use when masking options are
            "token_scores" or "regression"

        encoder_kwargs: dict
            Parameters for the continuous value encoder. There are three types of encoders available:

            1) "scale_adapt" (default): A frequency-based encoder using sinusoidal and/or sigmoid functions.
            Parameters include:
            - kind: "scale_adapt" - Specifies this encoder type
            - n_sin_basis: int, default=24
                Number of sinusoidal (sin, cos) basis function pairs to use.
                Higher values provide more resolution for distinguishing similar input values.
            - basis_scale: float, default=0.1
                Scaling factor for sine basis frequencies. Eg
                sin(x * basis_scale * ki), cos(x * basis_scale * ki),
                not applied to sigmoids (default is 1.0)
                - Lower values (e.g., 0.05): Better for encoding wide ranges of values
                - Higher values (e.g., 0.5): Better for encoding narrow ranges with high precision
            - shift: float, default=0.0
                Value subtracted from inputs before encoding.
                Set to the minimum expected value in your data to center the encoding around 0.
            - sigmoid_centers: list[float], default=None
                Center points for sigmoid functions where the sigmoid equals 0.5.
                Example: [0.5, 2.0, 5.0] creates features sensitive to these specific values.
            - sigmoid_orientations: list[float], default=None
                Controls the direction and steepness of each sigmoid function.
                Must match the length of sigmoid_centers.
            - trainable: bool, default=True
                If True, makes the sine basis parameters learnable during training.

            2) "mlp": A simple MLP encoder that does not require additional parameters.

            3) "mlp_with_special_tokens": An MLP encoder with special token handling.
            Parameters include:
            - kind: "mlp_with_special_tokens" - Specifies this encoder type
            - zero_as_special_token: bool
                If True, treats zero values as special tokens rather than continuous values.

            An example of parameters for ScaleAdapt:
            continuous_value_encoder_kwargs:
                kind: scale_adapt
                n_sin_basis: 11
                shift: 0.0
                basis_scale: 0.1
                sigmoid_centers: [0.0]
                sigmoid_orientations: [1.0]
                trainable: False


    """

    field_name: str
    vocab_size: int | None = None
    pretrained_embedding: PreTrainedEmbeddingConfig | None = None
    vocab_update_strategy: str = "static"
    is_masked: bool = False
    is_input: bool = True
    decode_modes: dict[str, dict] | None = None
    tokenization_strategy: str = "tokenize"
    num_special_tokens: int = 0
    encoder_kwargs: dict | None = None

    def __post_init__(self):
        if isinstance(self.decode_modes, list):
            self.decode_modes = {i: {} for i in self.decode_modes}
        if self.is_masked and not self.decode_modes:
            raise ValueError("Requested masking with no decode modes")

    @property
    def is_decode(self):
        return bool(self.decode_modes)

    def update_vocab_size(self, multifield_tokenizer):
        self.vocab_size = multifield_tokenizer.field_vocab_size(self.field_name)
        self.num_special_tokens = len(multifield_tokenizer.all_special_tokens)

    def to_dict(self):
        return asdict(self)

    def __setstate__(self, state):
        if "masked_output_modes" in state:
            state["decode_modes"] = state.pop("masked_output_modes")
        if "continuous_value_encoder_kwargs" in state:
            state["encoder_kwargs"] = state.pop("continuous_value_encoder_kwargs")
        self.__dict__.update(state)
        self.__post_init__()

    def update_pretrained_embedding_indices(self, multifield_tokenizer):
        (
            embedding_indices_to_freeze,
            pre_trained_indices_to_use,
        ) = self.pretrained_embedding.get_indices_of_pretrained_token_embeddings(
            self.field_name, multifield_tokenizer
        )
        self.pretrained_embedding.embedding_indices_to_freeze = (
            embedding_indices_to_freeze
        )
        self.pretrained_embedding.pre_trained_indices_to_use = (
            pre_trained_indices_to_use
        )


@dataclass(eq=True, repr=True)
class LabelColumnInfo:
    """
    Represents configuration settings for a column in a dataset that will be used as a label during training.

    This dataclass encapsulates properties related to a label column, such as its name,
    task type (classification or regression), and training-related parameters like stratification,
    domain adaptation, and shared layer configurations for grouped labels.

    Args:
    ----
        label_column_name (str): The name of the label column in the dataset.
        task_group (Optional[str], optional): A grouping identifier for related labels.
            If multiple label columns share the same `task_group`, an additional shared
            layer is created to learn a common representation for these labels. Defaults to None.
        is_stratification_label (bool, optional): If True, the label is used for stratified sampling
            during data splitting, ensuring balanced label representation in each split.
            Defaults to False.
        is_regression_label (bool, optional): If True, the label is treated as a regression target;
            otherwise, it is treated as a classification target. Defaults to False.
        classifier_depth (int, optional): The number of layers in the classifier head for this label.
            Must be a positive integer. Defaults to 1.
        gradient_reversal_coefficient (Optional[float], optional): The gradient reversal coefficient for
            domain adaptation, as described in https://arxiv.org/abs/1409.7495.
            Must be a positive float; starting with 0.1 is recommended to avoid excessive gradient scaling.
            Defaults to None.
        n_unique_values (Optional[int], optional): Number of unique labels. Required for classification
            tasks, can remain None for regression tasks.
        silent_label_values (Optional[list]): label values to be silently overlooked during training
            these label values will be set to -100 when passed to the model and will therefore not
            affect training. Useful for datasets that contain "Unknown" as a label value or similar.

    Attributes:
    ----------
        output_size (int): The number of unique labels for classification tasks or 1 for regression tasks.

    Raises:
    ------
        ValueError: If `output_size` is accessed before it is set for classification tasks (i.e., not regression).

    """

    label_column_name: str
    task_group: str | None = None
    is_stratification_label: bool = False
    is_regression_label: bool = False
    is_multilabel: bool = False
    classifier_depth: int = 1
    gradient_reversal_coefficient: float | None = None
    n_unique_values: int | None = None
    silent_label_values: list[str] | None = None
    multilabel_str_sep: str = "|"

    def __setstate__(self, state):
        # Handle legacy field name "output_size"
        if "output_size" in state:
            state["n_unique_values"] = state.pop("output_size")
        self.__dict__.update(state)

    @property
    def output_size(self) -> int:
        """
        The number of unique labels for this column.

        For regression tasks (`is_regression_label=True`), the output size is always 1.
        For classification tasks, the output size is the internally stored `_output_size`
        which must be set using `update_n_unique_values`.

        Returns
        -------
            int: The number of unique labels (1 for regression, or the number of classes for classification).

        Raises
        ------
            ValueError: If `output_size` is accessed before it is set for a classification task.

        """
        if self.is_regression_label:
            return 1
        elif self.n_unique_values is not None:
            return self.n_unique_values
        raise ValueError(
            "`n_unique_values` must be set, for non-regression classes. Run `update_n_unique_values` with valid label_dict!"
        )

    def update_n_unique_values(self, label_dict):
        """
        Updates the output size based on the provided label dictionary.

        The `label_dict` should map label column names to lists of unique label values. This method sets the
        output size to the length of the list corresponding to `label_column_name`.

        Args:
        ----
            label_dict (Dict[str, List]): A dictionary where keys are label column names and values are lists
                of unique label values.

        Raises:
        ------
            KeyError: If `label_column_name` is not found in `label_dict`.
        """
        self.n_unique_values = len(label_dict[self.label_column_name])

    def to_dict(self):
        """
        Converts the `LabelColumnInfo` instance to a dictionary, including the `output_size` property.

        The resulting dictionary includes all public attributes and the computed `output_size`.
        If `output_size` cannot be determined (e.g., not set for a classification task), it is set to None in the dictionary.

        Returns
        -------
            Dict: A dictionary representation of the `LabelColumnInfo` instance.

        """
        return asdict(self)
