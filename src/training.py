"""Model, data and checkpoint helpers shared by the worker processes."""

import os
from pathlib import Path

import torch
from torch.utils.data import default_collate


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch_for_step(dataset, step, rank, workers, batch_size):
    """Split a reproducible global batch of indexed classification samples."""
    generator = torch.Generator().manual_seed(1234 + step)
    size = workers * batch_size
    if size > len(dataset):
        raise ValueError("global batch size exceeds the dataset")
    indices = torch.randperm(len(dataset), generator=generator)[rank * batch_size:(rank + 1) * batch_size]
    return default_collate([dataset[index] for index in indices.tolist()])


def save_checkpoint(path, model, step, optimizer=None, metadata=None):
    """Publish a complete checkpoint with an atomic rename on the same filesystem."""
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save({"step": step, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "metadata": metadata}, temp)
    os.replace(temp, path)


def load_checkpoint(path, model, optimizer=None, expected_metadata=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata")
    if metadata is not None and expected_metadata is not None and metadata != expected_metadata:
        raise ValueError(f"checkpoint training configuration does not match this run: {metadata}")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        state = checkpoint.get("optimizer")
        if state is None:
            if optimizer.param_groups[0]["momentum"] != 0 and checkpoint["step"] > 0:
                raise ValueError("checkpoint has no optimizer state; cannot resume with momentum")
        else:
            saved = state["param_groups"][0]
            current = optimizer.param_groups[0]
            if any(saved[key] != current[key] for key in ("lr", "momentum")):
                raise ValueError("checkpoint learning rate or momentum does not match this run")
            optimizer.load_state_dict(state)
    return checkpoint["step"]
