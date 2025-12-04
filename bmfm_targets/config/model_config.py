from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from bmfm_targets.config import FieldInfo, LabelColumnInfo

logger = logging.get_logger(__name__)

PerformerKernel = Enum(
    "PerformerKernel", {name: name for name in ["cosh", "exp", "elu", "relu"]}, type=str
)
FeatureGenerationAlgorithm = Enum(
    "FeatureGenerationAlgorithm",
    {name: name for name in ["auto", "kacs", "qr", "guassian"]},
    type=str,
)


class ModelingStrategy(str, Enum):
    """
    Enumeration of all  modeling strategies.

    Members:
        MLM: mask language modeling
        SEQUENCE_CLASSIFICATION: sequence-level classification
        SEQUENCE_LABELING: token-level labeling
        MULTITASK: multitask learning
    """

    SEQUENCE_LABELING = "sequence_labeling"
    MLM = "mlm"
    MULTITASK = "multitask"
    SEQUENCE_CLASSIFICATION = "sequence_classification"


class SCModelConfigBase(PretrainedConfig):
    def to_dict(self):
        """Serializes class to a Python dictionary."""
        output = super().to_dict()
        if self.fields is not None:
            output["fields"] = [fi.to_dict() for fi in self.fields]
        else:
            output["fields"] = None

        if self.label_columns is not None:
            output["label_columns"] = [fi.to_dict() for fi in self.label_columns]
        else:
            output["label_columns"] = None
        output = {k: v.value if isinstance(v, Enum) else v for k, v in output.items()}
        return output

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs) -> "PretrainedConfig":
        fields = [FieldInfo(**fi) for fi in config_dict["fields"]]
        label_columns = (
            [LabelColumnInfo(**fi) for fi in config_dict["label_columns"]]
            if config_dict.get("label_columns")
            else None
        )
        config_dict["fields"] = fields
        config_dict["label_columns"] = label_columns
        return super().from_dict(config_dict, **kwargs)

    def __setstate__(self, state):
        state.setdefault("label_columns", None)
        state.setdefault("_output_attentions", state.get("output_attentions", None))
        self.__dict__.update(state)


class SCBertConfig(SCModelConfigBase):
    """
    Configuration class to store the configuration of a [`SCBertModel`].

    It is used to instantiate a BERT model according to the
    specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that
    of the BERT architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    the model outputs. Read the documentation from [`PretrainedConfig`]
    for more information.


    Attributes
    ----------
        fields (`list[FieldInfo]`):
            A list of `FieldInfo` objects, each representing a field.
        checkpoint (str): checkpoint to load from
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    """

    model_type = "scbert"

    def __init__(
        self,
        fields: list[FieldInfo] | None = None,
        label_columns: list[LabelColumnInfo] | None = None,
        checkpoint: str | None = None,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        use_cache: bool = True,
        classifier_dropout: float | None = None,
        max_position_embeddings: int = 512,
        position_embedding_type: str | None = None,
        attention: str | None = None,
        attention_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Args:
        ----
        fields (`list[FieldInfo]`):
        A list of `FieldInfo` objects, each representing a field.
        num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
        is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.

        """
        self.attention = attention
        self.attention_kwargs = attention_kwargs

        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.fields = fields
        self.label_columns = label_columns
        self.checkpoint = checkpoint
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        if classifier_dropout is not None:
            self.classifier_dropout = classifier_dropout
        else:
            self.classifier_dropout = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type


class SCModernBertConfig(SCModelConfigBase):
    """
    The configuration class to store the configuration of a [`ModernBertModel`]. It is used to instantiate an ModernBert
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ModernBERT-base.
    e.g. [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base).
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
    ----
        fields (`list[FieldInfo]`):
            A list of `FieldInfo` objects, each representing a field.
        checkpoint (torch.serialiaziation.FILE_LIKE):
            checkpoint to load from
        vocab_size (`int`, *optional*, defaults to 50368):
            Vocabulary size of the ModernBert model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ModernBertModel`]
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 1152):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 22):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer decoder.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder. Will default to `"gelu"`
            if not specified.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_cutoff_factor (`float`, *optional*, defaults to 2.0):
            The cutoff factor for the truncated_normal_initializer for initializing all weight matrices.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        norm_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the normalization layers.
        pad_token_id (`int`, *optional*, defaults to 50283):
            Padding token id.
        global_rope_theta (`float`, *optional*, defaults to 160000.0):
            The base period of the global RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        global_attn_every_n_layers (`int`, *optional*, defaults to 3):
            The number of layers between global attention layers.
        local_attention (`int`, *optional*, defaults to 128):
            The window size for local attention.
        local_rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the local RoPE embeddings.
        embedding_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the MLP layers.
        mlp_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the MLP layers.
        decoder_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the decoder layers.
        classifier_pooling (`str`, *optional*, defaults to `"cls"`):
            The pooling method for the classifier. Should be either `"cls"` or `"mean"`. In local attention layers, the
            CLS token doesn't attend to all tokens on long sequences.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the classifier.
        classifier_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the classifier.
        classifier_activation (`str`, *optional*, defaults to `"gelu"`):
            The activation function for the classifier.
        deterministic_flash_attn (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic flash attention. If `False`, inference will be faster but not deterministic.
        sparse_prediction (`bool`, *optional*, defaults to `False`):
            Whether to use sparse prediction for the masked language model instead of returning the full dense logits.
        sparse_pred_ignore_index (`int`, *optional*, defaults to -100):
            The index to ignore for the sparse prediction.
        reference_compile (`bool`, *optional*):
            Whether to compile the layers of the model which were compiled during pretraining. If `None`, then parts of
            the model will be compiled if 1) `triton` is installed, 2) the model is not on MPS, 3) the model is not
            shared between devices, and 4) the model is not resized after initialization. If `True`, then the model may
            be faster in some scenarios.
        repad_logits_with_grad (`bool`, *optional*, defaults to `False`):
            When True, ModernBertForMaskedLM keeps track of the logits' gradient when repadding for output. This only
            applies when using Flash Attention 2 with passed labels. Otherwise output logits always have a gradient.
        position_embedding_type (`str`, *optional*, defaults to None):
            Leave set to to None to use ModernBerts rotary emeddings.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.

    """

    model_type = "scmodernbert"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"rope_theta": "global_rope_theta"}

    def __init__(
        self,
        fields: list[FieldInfo] | None = None,
        label_columns: list[LabelColumnInfo] | None = None,
        checkpoint: str | None = None,
        vocab_size: int = 50368,
        hidden_size: int = 768,
        intermediate_size: int = 1152,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 12,
        hidden_activation: str = "gelu",
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        initializer_cutoff_factor: float = 2.0,
        norm_eps: float = 1e-5,
        norm_bias: bool = False,
        pad_token_id: int = 2,
        global_rope_theta: float = 160000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        global_attn_every_n_layers: int = 3,
        local_attention: int = 128,
        local_rope_theta: float = 10000.0,
        embedding_dropout: float = 0.0,
        mlp_bias: bool = False,
        mlp_dropout: float = 0.0,
        decoder_bias: bool = True,
        classifier_pooling: Literal["cls", "mean"] = "cls",
        classifier_dropout: float = 0.0,
        classifier_bias: bool = False,
        classifier_activation: str = "gelu",
        deterministic_flash_attn: bool = False,
        sparse_prediction: bool = False,
        sparse_pred_ignore_index: int = -100,
        reference_compile: bool | None = None,
        repad_logits_with_grad: bool = False,
        position_embedding_type: str | None = None,
        is_decoder: bool = False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.fields = fields
        self.checkpoint = checkpoint
        self.label_columns = label_columns
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.initializer_cutoff_factor = initializer_cutoff_factor
        self.norm_eps = norm_eps
        self.norm_bias = norm_bias
        self.global_rope_theta = global_rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.local_attention = local_attention
        self.local_rope_theta = local_rope_theta
        self.embedding_dropout = embedding_dropout
        self.mlp_bias = mlp_bias
        self.mlp_dropout = mlp_dropout
        self.decoder_bias = decoder_bias
        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.classifier_bias = classifier_bias
        self.classifier_activation = classifier_activation
        self.deterministic_flash_attn = deterministic_flash_attn
        self.sparse_prediction = sparse_prediction
        self.sparse_pred_ignore_index = sparse_pred_ignore_index
        self.reference_compile = reference_compile
        self.repad_logits_with_grad = repad_logits_with_grad

        # SCLayer-specfic parameters
        self.position_embedding_type = position_embedding_type
        self.layer_norm_eps = norm_eps
        self.hidden_dropout_prob = embedding_dropout
        self.hidden_act = hidden_activation
        self.is_decoder = is_decoder


class SCPerformerConfig(SCModelConfigBase):
    """
    Configuration class to store the configuration of a [`SCPerformerModel`].

    It is used to instantiate a BERT model according to the
    specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that
    of the BERT architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    the model outputs. Read the documentation from [`PretrainedConfig`]
    for more information.

    """

    model_type = "scperformer"

    def __init__(
        self,
        fields: list[FieldInfo] | None = None,
        label_columns: list[LabelColumnInfo] | None = None,
        checkpoint: str | None = None,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        use_cache: bool = True,
        kernel_type: str | Callable | PerformerKernel = PerformerKernel.exp,
        use_recurrent_decoding: bool = False,
        kernel_epsilon: float = 1e-4,
        normalize_output: bool = True,
        normalization_stabilizer: float = 1e-6,
        max_position_embeddings: int = 512,
        position_embedding_type: str | None = None,
        num_random_features: int | None = None,
        regularize_feature_norms: bool = True,
        causal: bool = False,
        feature_generation_algorithm: str
        | FeatureGenerationAlgorithm = (FeatureGenerationAlgorithm.qr),
        feature_redraw_interval: int | None = None,
        classifier_dropout: float | None = None,
        **kwargs,
    ):
        """
        Args:
        ----
        fields (`list[FieldInfo]`):
            A list of `FieldInfo` objects, each representing a field.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        kernel_type (`str` or `Callable` or `PerformerKernel`, *optional*, defaults to `PerformerKernel.exp`):
            The kernel to use for the attention layer. If a string, `"cosh"`, `"exp"`, `"elu"` and `"relu"` are supported.
        use_recurrent_decoding (`bool`, *optional*, defaults to `False`):
            Whether to use recurrent decoding or not.
        kernel_epsilon (`float`, *optional*, defaults to 1e-4):
            The epsilon to use for the kernel.
        normalize_output (`bool`, *optional*, defaults to `True`):
            Whether to normalize the output of the attention layer.
        normalization_stabilizer (`float`, *optional*, defaults to 1e-6):
            The stabilizer to use for normalization.
        use_linear_layers (`bool`, *optional*, defaults to `True`):
            Whether to use linear layers or not.
        linear_layer_names (`Sequence[str]`, *optional*, defaults to `("q_linear", "k_linear", "v_linear", "out_linear")`):
            The names of the linear layers.
        num_random_features (`int` or `None`, *optional*, defaults to `None`):
            The number of random features to use for the attention layer.
        use_thick_features (`bool`, *optional*, defaults to `False`):
            Whether to use thick features or not.
        regularize_feature_norms (`bool`, *optional*, defaults to `True`):
            Whether to regularize the feature norms or not.
        causal (`bool`, *optional*, defaults to `False`):
            Whether to use causal attention from `fast_transformers` or not.
        orthogonal_feature_algorithm (`str` or `OrthogonalFeatureAlgorithm`, *optional*, defaults to `OrthogonalFeatureAlgorithm.auto`):
            The algorithm to use for orthogonal features. If a string, `"auto"`, `"kacs"` and `"qr"` are supported.
        feature_redraw_interval (`int` or `None`, *optional*, defaults to `100`):
            The interval at which to redraw the features. If `None`, the features are not redrawn.

        """
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.fields = fields
        self.label_columns = label_columns
        self.checkpoint = checkpoint
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        if classifier_dropout is not None:
            self.classifier_dropout = classifier_dropout
        else:
            self.classifier_dropout = hidden_dropout_prob
        self.kernel_type = kernel_type
        self.use_recurrent_decoding = use_recurrent_decoding
        self.kernel_epsilon = kernel_epsilon
        self.normalize_output = normalize_output
        self.normalization_stabilizer = normalization_stabilizer
        self.num_random_features = num_random_features
        self.regularize_feature_norms = regularize_feature_norms
        self.causal = causal
        self.feature_generation_algorithm = feature_generation_algorithm
        self.feature_redraw_interval = feature_redraw_interval
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type


class SCNystromformerConfig(SCModelConfigBase):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
    ----
        vocab_size (`int`, *optional*, defaults to 30000):
            Vocabulary size of the Nystromformer model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`NystromformerModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`NystromformerModel`].
        segment_means_seq_len (`int`, *optional*, defaults to 64):
            Sequence length used in segment-means.
        num_landmarks (`int`, *optional*, defaults to 64):
            The number of landmark (or Nystrom) points to use in Nystrom approximation of the softmax self-attention
            matrix.
        conv_kernel_size (`int`, *optional*, defaults to 65):
            The kernel size of depthwise convolution used in Nystrom approximation.
        inverse_method (`str`, *optional*, defaults to `original`):
            Whether or not to use exact coefficient computation for the initial values for the iterative method of
            calculating the Moore-Penrose inverse of a matrix. Alternative iterative methods include newton and
            chebyshev.
        inverse_n_iter (`int`, *optional*, defaults to 6):
            Number of iterations to use to caculate the pseudoinverse of the matrix.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.

    Example:
    -------
    ```python
    >>> from transformers import NystromformerModel, NystromformerConfig

    >>> # Initializing a Nystromformer uw-madison/nystromformer-512 style configuration
    >>> configuration = NystromformerConfig()

    >>> # Initializing a model from the uw-madison/nystromformer-512 style configuration
    >>> model = NystromformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    """

    model_type = "scnystromformer"

    def __init__(
        self,
        fields: list[FieldInfo] | None = None,
        label_columns: list[LabelColumnInfo] | None = None,
        checkpoint: str | None = None,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu_new",
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        position_embedding_type: str | None = None,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        num_landmarks: int = 64,
        conv_kernel_size: int = 65,
        inverse_method: str = "original",
        inverse_n_iter: int = 6,
        pad_token_id: int = 2,
        use_cache: bool = True,
        classifier_dropout: float | None = None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.fields = fields
        self.label_columns = label_columns
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.num_landmarks = num_landmarks
        self.conv_kernel_size = conv_kernel_size
        self.inverse_method = inverse_method
        self.inverse_n_iter = inverse_n_iter
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type

        self.use_cache = use_cache
        if classifier_dropout is not None:
            self.classifier_dropout = classifier_dropout
        else:
            self.classifier_dropout = hidden_dropout_prob
        self.input_vocab_size = 0
        self.output_vocab_size = 0
        self.checkpoint = checkpoint
