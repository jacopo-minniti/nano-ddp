"""Centralized gradient exchange and supervised group restarts."""

import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback

import torch

from src.dashboard import Dashboard
from src.data import DATASETS, load_dataset
from src.distributed import (
    ParallelTrainer, apply_gradients, checkpoint_metadata, local_gradients,
    model_digest, optimizer_digest, optimizer_tensors, tensor_bytes,
)
from src.models import MODEL_TYPES, build_model
from src.training import choose_device, load_checkpoint, save_checkpoint


def emit(events, message):
    events.put(message)


def maybe_inject(config, rank, step, events):
    target = 0 if config.fault == "coordinator-crash" else 1
    if config.fault == "none" or rank != target or step not in (config.fault_steps or (config.fault_step,)):
        return
    # Earlier versions used a single marker; honor it for the original step.
    if Path(config.fault_marker).exists():
        log = Path(config.checkpoint).parent / "run.log"
        if log.exists() and any(line.split("FAULT ", 1)[-1].rstrip().endswith(f"step={step}")
                                for line in log.read_text().splitlines() if "FAULT " in line):
            return
    marker = Path(f"{config.fault_marker}.step-{step}")
    try:
        # The marker survives a restart, so this fault is injected only once.
        with marker.open("x"):
            pass
    except FileExistsError:
        return
    if config.fault == "slow-worker":
        emit(events, f"FAULT rank={rank} sleeping for {config.delay:.1f}s at step={step}")
        time.sleep(config.delay)
    else:
        emit(events, f"FAULT rank={rank} crashing at step={step}")
        # Flush the event before deliberately killing the process.
        time.sleep(0.1)
        os._exit(1)


def receive(connection, timeout, description):
    if not connection.poll(timeout):
        raise TimeoutError(f"timed out waiting for {description} after {timeout:g}s")
    try:
        return connection.recv()
    except EOFError as exc:
        raise ConnectionError(f"connection closed while waiting for {description}") from exc



def worker(rank, config, connection, events):
    torch.set_num_threads(1)
    try:
        device = choose_device(config.device)
        model = build_model(config.model, device, config.layers)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
        loaded = load_checkpoint(config.checkpoint, model, optimizer, checkpoint_metadata(config))
        dataset = load_dataset(config.dataset, config.data_dir)
        emit(events, f"READY rank={rank} pid={os.getpid()} device={device} checkpoint={loaded}")
        while True:
            command = receive(connection, max(60, config.timeout * 3), "coordinator command")
            if command[0] == "stop":
                return
            step = command[1]
            maybe_inject(config, rank, step, events)
            loss, gradients = local_gradients(model, dataset, config, rank, step, device)
            connection.send((loss, gradients))
            averaged = receive(connection, max(60, config.timeout * 3), "averaged gradients")
            apply_gradients(model, optimizer, averaged)
            connection.send(("updated", step, model_digest(model), optimizer_digest(optimizer)))
    except Exception as exc:
        emit(events, f"ERROR rank={rank}: {type(exc).__name__}: {exc}")
        emit(events, f"TRACEBACK {traceback.format_exc()!r}")
        raise SystemExit(1) from None
    finally:
        connection.close()


def coordinator(config, connections, events):
    torch.set_num_threads(1)
    try:
        device = choose_device(config.device)
        model = build_model(config.model, device, config.layers)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
        start = load_checkpoint(config.checkpoint, model, optimizer, checkpoint_metadata(config))
        dataset = load_dataset(config.dataset, config.data_dir)
        emit(events, f"READY rank=0 coordinator pid={os.getpid()} device={device} checkpoint={start}")
        model_size = tensor_bytes(model.parameters())
        state_size = tensor_bytes(optimizer_tensors(optimizer))
        emit(events, f"MEMORY step={start} per_replica_model_bytes={model_size} per_replica_optimizer_bytes={state_size} total_replicated_bytes={(model_size + state_size) * config.workers}")
        for step in range(start + 1, config.steps + 1):
            emit(events, f"STEP {step} started")
            for conn in connections:
                conn.send(("step", step))
            maybe_inject(config, 0, step, events)
            loss, gradients = local_gradients(model, dataset, config, 0, step, device)
            losses = [loss]
            for rank, conn in enumerate(connections, 1):
                other_loss, other_gradients = receive(conn, config.timeout, f"gradients from rank {rank}")
                losses.append(other_loss)
                for total, grad in zip(gradients, other_gradients, strict=True):
                    total += grad
            averaged = [gradient / config.workers for gradient in gradients]
            for conn in connections:
                conn.send(averaged)
            apply_gradients(model, optimizer, averaged)
            digest = model_digest(model)
            state_digest = optimizer_digest(optimizer)
            for rank, conn in enumerate(connections, 1):
                ack = receive(conn, config.timeout, f"update from rank {rank}")
                if ack != ("updated", step, digest, state_digest):
                    raise ValueError(f"rank {rank} has a different model or unexpected acknowledgement: {ack}")
            save_checkpoint(config.checkpoint, model, step, optimizer, checkpoint_metadata(config))
            state_size = tensor_bytes(optimizer_tensors(optimizer))
            emit(events, f"COMMIT step={step} loss={sum(losses) / len(losses):.4f} replicas={config.workers} model={digest} optimizer={state_digest}")
            emit(events, f"MEMORY step={step} per_replica_model_bytes={model_size} per_replica_optimizer_bytes={state_size} total_replicated_bytes={(model_size + state_size) * config.workers} checkpoint_file_bytes={Path(config.checkpoint).stat().st_size}")
        emit(events, f"COMPLETE step={config.steps}")
        for conn in connections:
            conn.send(("stop",))
    except Exception as exc:
        emit(events, f"ERROR coordinator: {type(exc).__name__}: {exc}")
        emit(events, f"TRACEBACK {traceback.format_exc()!r}")
        raise SystemExit(1) from None
    finally:
        for conn in connections:
            conn.close()



class DataParallelTrainer(ParallelTrainer):
    """Centralized, synchronous data parallel training with group restart."""

    def run(self):
        """Run centralized training, restarting the group after a failure."""
        config = self.config
        if config.model not in MODEL_TYPES:
            raise ValueError(f"unknown model {config.model!r}; available: {', '.join(MODEL_TYPES)}")
        if config.dataset not in DATASETS:
            raise ValueError(f"unknown dataset {config.dataset!r}; available: {', '.join(DATASETS)}")
        # Download once in the supervisor, before workers start and before the live display.
        print(f"Preparing {config.dataset} data (downloaded automatically if missing)...", flush=True)
        load_dataset(config.dataset, config.data_dir, download=True)
        with Dashboard(config, method="dp") as dashboard:
            self._train(dashboard)


    def _train(self, dashboard):
        config = self.config
        if config.max_restarts is None:
            config.max_restarts = max(2, len(set(config.fault_steps or (config.fault_step,))) if config.fault != "none" else 0)
        if config.layers < 1:
            raise ValueError("layers must be at least one")
        if config.workers < 2:
            raise ValueError("at least two workers are required")
        if config.steps < 1 or config.batch_size < 1 or config.timeout <= 0 or config.max_restarts < 0:
            raise ValueError("steps, batch size and timeout must be positive; max restarts cannot be negative")
        if config.fault != "none" and any(not 1 <= step <= config.steps for step in (config.fault_steps or (config.fault_step,))):
            raise ValueError("all fault steps must be within the training run")
        if config.lr <= 0 or not 0 <= config.momentum < 1:
            raise ValueError("learning rate must be positive and momentum must be in [0, 1)")
        checkpoint = Path(config.checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model = build_model(config.model, torch.device("cpu"), config.layers)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
        if not checkpoint.exists():
            save_checkpoint(checkpoint, model, 0, optimizer, checkpoint_metadata(config))
        else:
            load_checkpoint(checkpoint, model, optimizer, checkpoint_metadata(config))
        config.fault_marker = config.fault_marker or str(checkpoint) + ".fault"
        ctx = mp.get_context("spawn")
        for attempt in range(config.max_restarts + 1):
            events = ctx.Queue()
            pairs = [ctx.Pipe() for _ in range(config.workers - 1)]
            processes = [ctx.Process(target=coordinator, args=(config, [a for a, _ in pairs], events), name="coordinator")]
            processes += [ctx.Process(target=worker, args=(rank, config, pair[1], events), name=f"worker-{rank}") for rank, pair in enumerate(pairs, 1)]
            dashboard.record(f"GROUP attempt={attempt + 1} starting from checkpoint")
            for process in processes:
                process.start()
            for a, b in pairs:
                a.close()
                b.close()
            complete = False
            last_event = time.monotonic()
            failure = None
            try:
                while True:
                    try:
                        message = events.get(timeout=0.2)
                        dashboard.record(message)
                        last_event = time.monotonic()
                        if message.startswith("COMPLETE"):
                            complete = True
                            break
                    except queue.Empty:
                        dead = [p for p in processes if p.exitcode is not None]
                        if dead:
                            failure = ", ".join(f"{p.name} exit={p.exitcode}" for p in dead)
                            break
                        if time.monotonic() - last_event > max(60, config.timeout * 3):
                            failure = "group watchdog timed out"
                            break
                if not complete:
                    dashboard.record(f"RECOVERY {failure or 'group failed'}; stopping all ranks and discarding incomplete step")
            finally:
                for process in processes:
                    if not complete and process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()
                if not complete:
                    for rank, process in enumerate(processes):
                        dashboard.record(f"STOPPED rank={rank} pid={process.pid} exit={process.exitcode} after group failure")
                    restored = torch.load(config.checkpoint, map_location="cpu", weights_only=True)["step"]
                    dashboard.record(f"RESTORE checkpoint={restored} available for restart")
                events.close()
                events.join_thread()
            if complete:
                return
            if attempt == config.max_restarts:
                raise RuntimeError("restart limit reached")
