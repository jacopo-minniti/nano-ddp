import torch

from src.distributed import apply_gradients
from src.models import build_model
from src.training import batch_for_step, load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    model = build_model("mlp", torch.device("cpu"))
    with torch.no_grad():
        next(model.parameters()).add_(1)
    save_checkpoint(tmp_path / "checkpoint.pt", model, 7)
    restored = build_model("mlp", torch.device("cpu"))
    assert load_checkpoint(tmp_path / "checkpoint.pt", restored) == 7
    for original, copy in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(original, copy)


def test_batches_are_disjoint_and_reproducible():
    class Dataset:
        data = torch.arange(100).reshape(100, 1, 1)
        targets = torch.arange(100)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return self.data[index], self.targets[index]

    dataset = Dataset()
    _, first = batch_for_step(dataset, 1, 0, 2, 10)
    _, second = batch_for_step(dataset, 1, 1, 2, 10)
    assert set(first.tolist()).isdisjoint(second.tolist())
    assert torch.equal(first, batch_for_step(dataset, 1, 0, 2, 10)[1])


def test_averaged_gradient_matches_combined_batch():
    torch.manual_seed(0)
    inputs = torch.randn(8, 3)
    targets = torch.randn(8, 1)
    base = torch.nn.Linear(3, 1)
    replicas = [torch.nn.Linear(3, 1) for _ in range(2)]
    for replica in replicas:
        replica.load_state_dict(base.state_dict())
    torch.nn.functional.mse_loss(base(inputs), targets).backward()
    for rank, replica in enumerate(replicas):
        torch.nn.functional.mse_loss(replica(inputs[rank * 4:(rank + 1) * 4]), targets[rank * 4:(rank + 1) * 4]).backward()
    average = [(a.grad.detach().numpy() + b.grad.detach().numpy()) / 2 for a, b in zip(*[list(r.parameters()) for r in replicas], strict=True)]
    apply_gradients(replicas[0], torch.optim.SGD(replicas[0].parameters(), lr=0.1), average)
    with torch.no_grad():
        for p in base.parameters():
            p.add_(p.grad, alpha=-0.1)
    for expected, actual in zip(base.parameters(), replicas[0].parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_momentum_checkpoint_restores_optimizer_state(tmp_path):
    from src.distributed import optimizer_tensors, tensor_bytes

    model = build_model("mlp", torch.device("cpu"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    assert tensor_bytes(optimizer_tensors(optimizer)) == 0
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert tensor_bytes(optimizer_tensors(optimizer)) == tensor_bytes(model.parameters())
    path = tmp_path / "momentum.pt"
    save_checkpoint(path, model, 1, optimizer)
    restored = build_model("mlp", torch.device("cpu"))
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9)
    assert load_checkpoint(path, restored, restored_optimizer) == 1
    for expected, actual in zip(optimizer_tensors(optimizer), optimizer_tensors(restored_optimizer), strict=True):
        torch.testing.assert_close(expected, actual)


def test_mlp_layer_counts():
    import pytest
    from torch import nn
    from src.models.mlp import MLP

    for count in (1, 2, 4):
        model = MLP(num_layers=count)
        assert sum(isinstance(layer, nn.Linear) for layer in model.layers) == count
        assert sum(isinstance(layer, nn.ReLU) for layer in model.layers) == count - 1
        assert model(torch.zeros(3, 1, 28, 28)).shape == (3, 10)
    assert sum(isinstance(layer, nn.Linear) for layer in MLP().layers) == 2
    with pytest.raises(ValueError):
        MLP(num_layers=0)


def test_dashboard_keeps_failure_history_after_resume(tmp_path):
    from src.dashboard import Dashboard
    from src.distributed import Config

    config = Config(data_dir="unused", checkpoint=str(tmp_path / "checkpoint.pt"))
    dashboard = Dashboard(config)
    dashboard.record("GROUP attempt=1 starting from checkpoint")
    for step in range(1, 8):
        dashboard.record(f"STEP {step} started")
        dashboard.record(f"FAULT rank=1 crashing at step={step}")
        dashboard.record("RECOVERY worker-1 exit=1; stopping all ranks")
        dashboard.record("STOPPED rank=1 pid=123 exit=1 after group failure")
        dashboard.record(f"RESTORE checkpoint={step - 1} available for restart")
    dashboard.log.close()
    restored = Dashboard(config)
    assert len(restored.failures) == 7
    assert restored.failures[0]["step"] == 1
    assert "STOPPED rank=1" in " ".join(restored.failures[0]["details"])
    assert restored.failures[-1]["step"] == 7
    restored.log.close()


def test_selection_uses_registered_implementations(monkeypatch, tmp_path):
    import pytest
    from torch import nn
    from src.data import DATASETS, load_dataset
    from src.models import MODEL_TYPES, build_model

    class OtherModel(nn.Module):
        def __init__(self, num_layers):
            super().__init__()
            self.layer = nn.Linear(num_layers, 1)

    dataset = [(torch.ones(3), 1)]
    monkeypatch.setitem(MODEL_TYPES, "other", OtherModel)
    monkeypatch.setitem(DATASETS, "other", lambda path, download: dataset)
    assert isinstance(build_model("other", torch.device("cpu"), num_layers=3), OtherModel)
    assert load_dataset("other", str(tmp_path)) is dataset
    with pytest.raises(ValueError, match="unknown model"):
        build_model("missing", torch.device("cpu"))
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("missing", str(tmp_path))


def test_unimplemented_distributed_components_are_explicit():
    import pytest
    from src.ddp import AllReduce, NanoDDP, RingAllReduce
    from src.models.nanogpt2 import NanoGPT2

    with pytest.raises(TypeError):
        AllReduce()
    with pytest.raises(NotImplementedError, match="Ring AllReduce"):
        RingAllReduce()
    with pytest.raises(NotImplementedError, match="NanoDDP"):
        NanoDDP(None).run()
    with pytest.raises(NotImplementedError, match="NanoGPT2"):
        NanoGPT2()


def test_checkpoint_rejects_different_workload(tmp_path):
    import pytest
    model = build_model("mlp", torch.device("cpu"))
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, 1, metadata={"dataset": "mnist"})
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(path, model, expected_metadata={"dataset": "other"})


def test_parallel_trainers_share_the_same_entry_point():
    import pytest
    from src.ddp import NanoDDP
    from src.distributed import ParallelTrainer
    from src.dp import DataParallelTrainer

    assert issubclass(DataParallelTrainer, ParallelTrainer)
    assert issubclass(NanoDDP, ParallelTrainer)
    with pytest.raises(TypeError):
        ParallelTrainer(None)


def test_resume_requires_checkpoint_metadata_and_optimizer(tmp_path):
    import pytest

    model = build_model("mlp", torch.device("cpu"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, 1)

    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(path, model, expected_metadata={"dataset": "mnist"})
    with pytest.raises(ValueError, match="no optimizer state"):
        load_checkpoint(path, model, optimizer=optimizer)
