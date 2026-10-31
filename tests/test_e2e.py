"""CPU-only MNIST runs that can also be used as reproducible demos."""

import os
from pathlib import Path

import pytest
import torch

from src.distributed import Config
from src.dp import DataParallelTrainer
from src.data import load_dataset


@pytest.fixture(scope="session")
def mnist_dir():
    path = Path(os.environ.get("NANODDP_DATA_DIR", "data"))
    load_dataset("mnist", str(path), download=True)
    return str(path.resolve())


@pytest.fixture(scope="session")
def baseline(mnist_dir, tmp_path_factory):
    checkpoint = tmp_path_factory.mktemp("baseline") / "checkpoint.pt"
    DataParallelTrainer(Config(data_dir=mnist_dir, checkpoint=str(checkpoint), workers=2, layers=3, steps=3,
                 batch_size=8, timeout=4, device="cpu", momentum=0.9)).run()
    return torch.load(checkpoint, map_location="cpu", weights_only=True)


@pytest.mark.e2e
@pytest.mark.parametrize("fault", ["worker-crash", "coordinator-crash", "slow-worker"])
def test_training_recovers_from_fault(mnist_dir, baseline, tmp_path, fault, capfd):
    checkpoint = tmp_path / "checkpoint.pt"
    config = Config(data_dir=mnist_dir, checkpoint=str(checkpoint), workers=2, layers=3, steps=3,
                    batch_size=8, timeout=4, device="cpu", momentum=0.9, fault=fault, fault_step=2,
                    fault_steps=(1, 2, 3) if fault == "worker-crash" else (), delay=8)
    DataParallelTrainer(config).run()
    output = capfd.readouterr().out
    print(output, end="")
    log = (tmp_path / "run.log").read_text()
    for filename in ("training.png", "failures.png", "predictions.png"):
        assert (tmp_path / filename).stat().st_size > 0
    assert "COMMIT step=" not in output
    assert "COMMIT step=3" in log
    assert "RECOVERY" in output
    if fault == "worker-crash":
        assert log.count("FAULT rank=1 crashing") == 3
        assert "RESTORE checkpoint=0" in log
        assert "RESTORE checkpoint=1" in log
        assert "RESTORE checkpoint=2" in log
        assert "STOPPED rank=0" in log and "STOPPED rank=1" in log
    assert "checkpoint=1" in log
    assert "COMPLETE step=3" in output
    result = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert result["step"] == 3
    for key in baseline["model"]:
        torch.testing.assert_close(result["model"][key], baseline["model"][key], rtol=0, atol=0)
    assert result["optimizer"]["state"]
    for key, state in baseline["optimizer"]["state"].items():
        torch.testing.assert_close(result["optimizer"]["state"][key]["momentum_buffer"], state["momentum_buffer"], rtol=0, atol=0)
    assert "per_replica_optimizer_bytes=0" in log
    assert "per_replica_optimizer_bytes=220200" in log
