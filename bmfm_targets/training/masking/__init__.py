"""Objects and methods related to masking of inputs."""

from .strategy import MaskingStrategy
from .masking import (
    Masker,
    prevent_attention_to_masked,
)

__all__ = [
    "Masker",
    "MaskingStrategy",
    "prevent_attention_to_masked",
]
