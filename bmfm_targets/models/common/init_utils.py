"""Utilities for safe weight initialization under transformers v5."""

from collections.abc import Callable

import torch.nn as nn


def init_unless_loaded(param: nn.Parameter | None, init_fn: Callable) -> None:
    """
    Run ``init_fn(param)`` only if ``param`` was not loaded from a checkpoint.

    transformers v5 calls ``_init_weights`` on every module during
    ``from_pretrained``, including modules whose weights were just loaded from
    the state dict.  Loaded params carry a ``_is_hf_initialized`` flag set by
    the HF loading machinery; honoring it prevents silently overwriting
    checkpoint weights with fresh random init.

    Args:
    ----
        param: The parameter tensor to (conditionally) initialize.  If
            ``None`` the function is a no-op.
        init_fn: A callable that takes no arguments and performs the in-place
            initialization (e.g. a lambda capturing the tensor).
    """
    if param is not None and not getattr(param, "_is_hf_initialized", False):
        init_fn()
