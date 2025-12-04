import os
from pathlib import Path

import torch


def get_label_dict(ckpt_path: Path | str) -> dict:
    device = check_gpu()

    ckpt = torch.load(
        ckpt_path,
        map_location=torch.device(device),
        weights_only=False,
    )

    label_dict = ckpt["hyper_parameters"]["label_dict"]
    return label_dict


def check_gpu(set_gpu: str | None = None) -> str:
    if set_gpu is None:
        if torch.cuda.is_available():
            distributed = ("NODE_RANK" in os.environ) and ("LOCAL_RANK" in os.environ)
            if distributed:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                device = torch.device(f"cuda:{local_rank}")
            else:
                device = torch.device("cuda")
            print("Using GPU")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(set_gpu)

    return device.type
