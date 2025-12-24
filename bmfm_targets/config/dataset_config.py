from enum import Enum


class SplitEnum(str, Enum):
    """Dataset split."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class ExposeZerosMode(str, Enum):
    """
    Defines how zero-valued (non-expressed) genes are exposed when read from dataset,
    typically the control expression in a paired perturbation dataset.

    This mode controls which input genes are retained before downstream heuristics
    (e.g., `pad_zero_expression_strategy`) are applied. Filtering occurs per sample
    before tokenization, without changing final input length (padding handles that).

    Modes:
    ------
    LABEL_NONZEROS:
        Keep zero-valued input genes only if their corresponding label (e.g., perturbed
        expression) is nonzero.
    NO_ZEROS:
        Exclude all zero-valued input genes entirely. Assumes non-expressed inputs
        carry no useful signal. For perturbations, control nonzero only.
    ALTERNATE:
        Used only with the `PerturbXDataLoader`. Alternates between `NO_ZEROS` and
        `LABEL_NONZEROS` each batch to encourage robustness.
    ALL:
        Include all genes in the input regardless of expression. Use this to delegate
        zero filtering to `pad_zero_expression_strategy` or other batch-level controls.
    """

    LABEL_NONZEROS = "label_nonzeros"
    NO_ZEROS = "no_zeros"
    ALTERNATE = "alternate"
    ALL = "all"
