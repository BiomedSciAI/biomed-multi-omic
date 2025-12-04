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
        prepend_tokens (list(str)) : modified prepend tokens for multiple CLS-like tokens.
            These tokens must be present in the original tokenizers vocab for special tokens
    """

    identifier: str = "all_genes"
    prepend_tokens: list[str] | None = None


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


@dataclass
class DatastoreConfig:
    """
    Configuration for datastore-based features (e.g., epigenetic annotations).

    Attributes
    ----------
    path : str
        Path to a Parquet file produced by scripts under `datasets.epigenetics`.
    bio_context_column_in_datastore : str
        Name of the column identifying the biological context (e.g., biosample).
    token_index_column : str
        Name of the column indexing tokens (typically `gene_symbol`).
    log1p_transform : bool
        Whether to apply a log1p transformation to feature values when loading.
    """

    path: str
    bio_context_column_in_datastore: str = "biosample_name"
    token_index_column: str = "gene_symbol"
    log1p_transform: bool = False


@dataclass(eq=True, repr=True)
class FieldInfo:
    """
    Represents information about a field.

    A "field" represents a type of data that is turned into a sequence of tokens and turned
    into a sequence of embeddings for model input. For example, fields can be "genes", "expressions",
    "dna_chunks" or condition tokens such as "perturbations".

    The definition of the field affects the data module collation, the structure of the model
    (because the layers that decode fields must be correctly configured), and the training module
    with metric calculations etc.

    Some fields are not used as inputs to the model, but rather as labels for training. This arises
    when training perturbation predictions where the label_expressions are the post-perturbation
    expressions, and the input is the pre-perturbation expressions. In this case,
    `is_input` is set to False, and the field is used for sequence labeling.

    Args:
    ----
        field_name (str): the name of the field
        vocab_size (int): The size of the vocabulary. This depends on the tokenizer and
          may not be known at instantiation time.
        vocab_update_strategy (str): The strategy to use for updating the vocabulary. Defaults to "static".
            If "dynamic", the vocabulary will be read from the dataset and the tokenizer will be updated
            before beginning training.
        pretrained_embedding (PreTrainedEmbeddingConfig): The pre trained embedding object, with the path to the embeddings.
        is_masked (bool): Whether to mask the field
        tokenization_strategy (str): The strategy to use for tokenization. Defaults to "tokenize".
          Other option is "continuous_value_encoder".
        is_input (bool): Whether the field is an input field. Defaults to True. Non-input fields
          are used for sequence labeling tasks or MLM with noise applied to inputs such as RDA.
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
            encoder_kwargs:
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
    datastore_config: DatastoreConfig | None = None

    def __post_init__(self):
        if isinstance(self.decode_modes, list):
            self.decode_modes = {i: {} for i in self.decode_modes}
        if self.is_masked and not self.decode_modes:
            raise ValueError("Requested masking with no decode modes")

    @property
    def is_decode(self):
        return bool(self.decode_modes)

    def update_vocab_size(self, multifield_tokenizer):
        if self.field_name in multifield_tokenizer.tokenizers:
            self.vocab_size = multifield_tokenizer.field_vocab_size(self.field_name)
        elif self.tokenization_strategy == "tokenize":
            raise ValueError(
                f"Must use a tokenizer or a continuous value encoder for field {self.field_name}."
            )

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
    Configuration for a dataset column used as a label during model training.

    This class defines metadata and training behavior for a specific label column,
    including its task type (classification or regression), grouping, stratification,
    and optional domain adaptation settings. It also supports multi-label and ontology-based
    configurations.

    Parameters
    ----------
    label_column_name : str
        Name of the label column in the dataset.
    task_group : str, optional
        Identifier for grouping related label columns. If multiple columns share
        the same `task_group`, a shared layer is added to learn a common representation.
        Default is None.
    is_stratification_label : bool, optional
        Whether this label is used for stratified sampling during data splitting.
        Default is False.
    is_regression_label : bool, optional
        Whether the label is treated as a regression target (otherwise classification).
        Default is False.
    is_multilabel : bool, optional
        Whether the column contains multiple labels per sample, separated by a delimiter.
        Default is False.
    is_bio_context_for_datastore : bool, optional
        Indicates whether this column defines a biological context for a datastore field.
        Default is False.
    classifier_depth : int, optional
        Number of layers in the classifier head for this label. Must be positive.
        Default is 1.
    gradient_reversal_coefficient : float, optional
        Coefficient for gradient reversal in domain adaptation, as described in
        [Ganin et al. (2015)](https://arxiv.org/abs/1409.7495). A small value (e.g. 0.1)
        is recommended to avoid excessive gradient scaling.
        Default is None.
    n_unique_values : int, optional
        Number of unique label values. Required for classification tasks, ignored for regression.
        Default is None.
    silent_label_values : list of str, optional
        Label values to be ignored during training. These are replaced with `-100` in model inputs
        and excluded from loss computation. Typically used for values like `"Unknown"`.
        Default is None.
    multilabel_str_sep : str, optional
        Delimiter for multi-label strings if `is_multilabel=True`. Default is `"|"`.
    label_ontology : str, optional
        Name of the ontology associated with this label column. Ontologies are stored in
        subfolders under `datasets/cell_ontology`. Default is None.

    Attributes
    ----------
    output_size : int
        The number of output units for this label — 1 for regression, or the number of unique
        classes for classification.

    Raises
    ------
    ValueError
        If `output_size` is accessed for a classification label before `n_unique_values`
        has been set.
    """

    label_column_name: str
    task_group: str | None = None
    is_stratification_label: bool = False
    is_regression_label: bool = False
    is_multilabel: bool = False
    is_bio_context_for_datastore: bool = False
    classifier_depth: int = 1
    gradient_reversal_coefficient: float | None = None
    n_unique_values: int | None = None
    silent_label_values: list[str] | None = None
    multilabel_str_sep: str = "|"
    label_ontology: str | None = None
    decode_from: int | None = None

    def __setstate__(self, state):
        """
        Handle backward compatibility for legacy field names.

        Converts legacy `"output_size"` field to `"n_unique_values"` on unpickling.
        """
        if "output_size" in state:
            state["n_unique_values"] = state.pop("output_size")
        self.__dict__.update(state)

    @property
    def output_size(self) -> int:
        """
        Number of output units for this label.

        Returns
        -------
        int
            `1` for regression labels, or the number of unique classes for classification.

        Raises
        ------
        ValueError
            If called before `n_unique_values` is set for a classification label.
        """
        if self.is_regression_label:
            return 1
        elif self.n_unique_values is not None:
            return self.n_unique_values
        raise ValueError(
            "`n_unique_values` must be set for non-regression labels. "
            "Call `update_n_unique_values(label_dict)` first."
        )

    def update_n_unique_values(self, label_dict: dict[str, list]):
        """
        Update the number of unique labels for this column.

        Parameters
        ----------
        label_dict : dict[str, list]
            Mapping of label column names to their unique label values.
            The length of the list for this column determines `n_unique_values`.

        Raises
        ------
        KeyError
            If `label_column_name` is not found in `label_dict`.
        """
        self.n_unique_values = len(label_dict[self.label_column_name])

    def to_dict(self) -> dict:
        """
        Convert this configuration object to a dictionary.

        Includes all public attributes and the computed `output_size` value,
        if available.

        Returns
        -------
        dict
            A dictionary representation of the configuration.
        """
        return asdict(self)
