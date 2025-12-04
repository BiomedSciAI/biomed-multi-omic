from enum import Enum


class AttentionKind(str, Enum):
    """
    Enumeration of all  attention kinds.

    Members:
        TORCH: torch sdp
        FLEX: CUDA flex attention
    """

    TORCH = "torch"
    FLEX = "flex"
