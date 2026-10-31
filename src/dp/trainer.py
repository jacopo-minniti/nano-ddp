"""Start and supervise the centralized data-parallel process group."""

import multiprocessing as mp
from pathlib import Path
import queue
import time

import torch

from src.dashboard import Dashboard
from src.data import DATASETS, load_dataset
from src.distributed import ParallelTrainer, checkpoint_metadata
from src.dp.processes import coordinator, worker
from src.models import MODEL_TYPES, build_model
from src.training import load_checkpoint, save_checkpoint


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
        """Supervise the group and restart it from the last checkpoint on failure."""
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
            processes += [
                ctx.Process(target=worker, args=(rank, config, pair[1], events), name=f"worker-{rank}")
                for rank, pair in enumerate(pairs, 1)
            ]

            dashboard.record(f"GROUP attempt={attempt + 1} starting from checkpoint")
            for process in processes: process.start()
            
            # Only the child processes need the pipe endpoints after startup.
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
                # On failure, stop the whole group before restarting from the checkpoint.
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

            if complete: return
            if attempt == config.max_restarts: raise RuntimeError("restart limit reached")
