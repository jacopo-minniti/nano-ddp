"""Model selection. Add new model constructors to MODEL_TYPES."""

import torch

from src.models.mlp import MLP

MODEL_TYPES = {"mlp": MLP}


def build_model(name, device, num_layers=2):
    try:
        model_type = MODEL_TYPES[name]
    except KeyError as exc:
        raise ValueError(f"unknown model {name!r}; available: {', '.join(MODEL_TYPES)}") from exc
    # Every replica must start with the same parameters.
    torch.manual_seed(0)
    return model_type(num_layers=num_layers).to(device)
