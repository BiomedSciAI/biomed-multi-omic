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
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(set_gpu)

    return device.type
