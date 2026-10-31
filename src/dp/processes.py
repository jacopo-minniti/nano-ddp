"""Coordinator and worker processes for centralized data parallelism."""

import os
from pathlib import Path
import traceback

import torch

from src.data import load_dataset
from src.distributed import (
    apply_gradients,
    checkpoint_metadata,
    local_gradients,
    model_digest,
    optimizer_digest,
    optimizer_tensors,
    tensor_bytes,
)
from src.dp.faults import maybe_inject
from src.models import build_model
from src.training import choose_device, load_checkpoint, save_checkpoint


def receive(connection, timeout, description):
    """Receive a message without waiting indefinitely for the other process."""
    if not connection.poll(timeout):
        raise TimeoutError(f"timed out waiting for {description} after {timeout:g}s")
    try:
        return connection.recv()
    except EOFError as exc:
        raise ConnectionError(f"connection closed while waiting for {description}") from exc


def worker(rank, config, connection, events):
    """Compute local gradients and apply the coordinator’s average."""
    torch.set_num_threads(1)

    try:
        device = choose_device(config.device)
        model = build_model(config.model, device, config.layers)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

        loaded = load_checkpoint(config.checkpoint, model, optimizer, checkpoint_metadata(config))
        dataset = load_dataset(config.dataset, config.data_dir)
        events.put(f"READY rank={rank} pid={os.getpid()} device={device} checkpoint={loaded}")

        while True:
            command = receive(connection, max(60, config.timeout * 3), "coordinator command")
            if command[0] == "stop": return

            step = command[1]
            maybe_inject(config, rank, step, events)
            loss, gradients = local_gradients(model, dataset, config, rank, step, device)
            connection.send((loss, gradients))
            averaged = receive(connection, max(60, config.timeout * 3), "averaged gradients")
            apply_gradients(model, optimizer, averaged)
            connection.send(("updated", step, model_digest(model), optimizer_digest(optimizer)))

    except Exception as exc:
        events.put(f"ERROR rank={rank}: {type(exc).__name__}: {exc}")
        events.put(f"TRACEBACK {traceback.format_exc()!r}")
        raise SystemExit(1) from None

    finally: connection.close()


def coordinator(config, connections, events):
    """Train as rank 0 and coordinate each synchronous update."""
    torch.set_num_threads(1)
    try:
        device = choose_device(config.device)
        model = build_model(config.model, device, config.layers)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

        start = load_checkpoint(config.checkpoint, model, optimizer, checkpoint_metadata(config))
        dataset = load_dataset(config.dataset, config.data_dir)
        events.put(f"READY rank=0 coordinator pid={os.getpid()} device={device} checkpoint={start}")

        model_size = tensor_bytes(model.parameters())
        state_size = tensor_bytes(optimizer_tensors(optimizer))

        events.put(f"MEMORY step={start} per_replica_model_bytes={model_size} per_replica_optimizer_bytes={state_size} total_replicated_bytes={(model_size + state_size) * config.workers}")

        for step in range(start + 1, config.steps + 1):
            events.put(f"STEP {step} started")
            for conn in connections: conn.send(("step", step))

            maybe_inject(config, 0, step, events)
            loss, gradients = local_gradients(model, dataset, config, 0, step, device)
            losses = [loss]

            # Rank 0 contributes its own gradients, then averages all ranks.
            for rank, conn in enumerate(connections, 1):
                other_loss, other_gradients = receive(conn, config.timeout, f"gradients from rank {rank}")
                losses.append(other_loss)
                for total, grad in zip(gradients, other_gradients, strict=True):
                    total += grad
            averaged = [gradient / config.workers for gradient in gradients]

            for conn in connections: conn.send(averaged)
            apply_gradients(model, optimizer, averaged)

            # Commit only after every replica confirms the same updated state.
            digest = model_digest(model)
            state_digest = optimizer_digest(optimizer)
            for rank, conn in enumerate(connections, 1):
                ack = receive(conn, config.timeout, f"update from rank {rank}")
                if ack != ("updated", step, digest, state_digest):
                    raise ValueError(f"rank {rank} has a different model or unexpected acknowledgement: {ack}")

            save_checkpoint(config.checkpoint, model, step, optimizer, checkpoint_metadata(config))
            state_size = tensor_bytes(optimizer_tensors(optimizer))
            
            events.put(f"COMMIT step={step} loss={sum(losses) / len(losses):.4f} replicas={config.workers} model={digest} optimizer={state_digest}")
            events.put(f"MEMORY step={step} per_replica_model_bytes={model_size} per_replica_optimizer_bytes={state_size} total_replicated_bytes={(model_size + state_size) * config.workers} checkpoint_file_bytes={Path(config.checkpoint).stat().st_size}")

        events.put(f"COMPLETE step={config.steps}")
        for conn in connections: conn.send(("stop",))

    except Exception as exc:
        events.put(f"ERROR coordinator: {type(exc).__name__}: {exc}")
        events.put(f"TRACEBACK {traceback.format_exc()!r}")
        raise SystemExit(1) from None

    finally:
        for conn in connections:
            conn.close()

