"""Shared configuration, trainer interface, and training helpers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib

import torch
import torch.nn.functional as F

from src.training import batch_for_step


@dataclass
class Config:
    data_dir: str
    checkpoint: str
    workers: int = 3
    model: str = "mlp"
    dataset: str = "mnist"
    layers: int = 2
    steps: int = 10
    batch_size: int = 32
    lr: float = 0.1
    momentum: float = 0.0
    timeout: float = 10.0
    device: str = "auto"
    max_restarts: int | None = None
    fault: str = "none"  # none, worker-crash, coordinator-crash, slow-worker
    fault_step: int = 2
    fault_steps: tuple[int, ...] = ()
    delay: float = 20.0
    fault_marker: str = ""


def checkpoint_metadata(config):
    return {
        "model": config.model,
        "dataset": config.dataset,
        "layers": config.layers,
        "workers": config.workers, 
        "batch_size": config.batch_size
    }



class ParallelTrainer(ABC):
    """Common entry point for a distributed training strategy."""

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def run(self):
        """Start training according to this strategy."""
        raise NotImplementedError


def apply_gradients(model, optimizer, gradients):
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        parameter.grad = torch.as_tensor(gradient, device=parameter.device)
    optimizer.step()


def optimizer_tensors(optimizer):
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            for value in optimizer.state.get(parameter, {}).values():
                if isinstance(value, torch.Tensor):
                    yield value


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def optimizer_digest(optimizer):
    digest = hashlib.sha256()
    for tensor in optimizer_tensors(optimizer):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:12]


def model_digest(model):
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:12]


def local_gradients(model, dataset, config, rank, step, device):
    model.zero_grad(set_to_none=True)
    images, labels = batch_for_step(dataset, step, rank, config.workers, config.batch_size)
    loss = F.cross_entropy(model(images.to(device)), labels.to(device))
    loss.backward()
    return float(loss.item()), [p.grad.detach().cpu().numpy().copy() for p in model.parameters()]

