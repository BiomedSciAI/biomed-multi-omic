"""Functions for serializing models to and from checkpoints."""

import os

import torch
import torch.distributed


def get_compute_device():
    if torch.cuda.is_available():
        distributed = ("NODE_RANK" in os.environ) and ("LOCAL_RANK" in os.environ)
        if distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def prepare_model_dict_from_checkpoint(
    checkpoint, base_model_prefix: str | None = None
):
    device = get_compute_device()
    model_dict = torch.load(checkpoint, map_location=device, weights_only=False)
    # manage loading model from lightening checkpoint
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
        for key in list(state_dict.keys()):
            new_key = key.removeprefix("model.")
            if "base_model" in new_key and base_model_prefix:
                new_key = new_key.replace("base_model", base_model_prefix)
            state_dict[new_key] = state_dict.pop(key)
        model_dict = state_dict
    return model_dict


def serialize_dataloader_states(state, device):
    """Serialize and collect dataloader states on rank 0."""
    distributed = ("NODE_RANK" in os.environ) and ("LOCAL_RANK" in os.environ)
    if distributed and (
        not torch.distributed.is_available() or not torch.distributed.is_initialized()
    ):
        raise RuntimeError(
            "Torch distributed is not available to collect dataloader states"
        )

    if not distributed or torch.distributed.get_world_size() == 1:
        state = {"world_size": 1, "state": [state]}
        return state

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    states = [None] * world_size if rank == 0 else None
    torch.distributed.gather_object(state, states)

    if rank != 0:
        return None

    return_state = {"world_size": world_size, "state": states}
    return return_state


def deserialize_dataloader_states(state):
    """Deserialization of dataloader state, see serialize_dataloader_states."""
    distributed = ("NODE_RANK" in os.environ) and ("LOCAL_RANK" in os.environ)
    checkpoint_world_size = state["world_size"]
    state = state["state"]
    if checkpoint_world_size == 1 and not distributed:
        return state[0]
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            f"Dataloader checkpoints require the same number of GPUs: {checkpoint_world_size}."
        )

    world_size = torch.distributed.get_world_size()
    if world_size != checkpoint_world_size:
        raise RuntimeError(
            f"Dataloader checkpoints require the same number of GPUs. You have {checkpoint_world_size} vs {world_size}"
        )

    rank = torch.distributed.get_rank()
    return state[rank]
