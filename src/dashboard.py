"""Terminal display, run log, and plots for a training run."""

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.visualizations import plot_failures, plot_predictions


def fields(message): return dict(re.findall(r"(\w+)=([^\s]+)", message))

def kib(value): return f"{int(value) / 1024:.1f} KiB"


class Dashboard:
    def __init__(self, config, method="dp"):
        self.config = config
        self.method = method
        self.directory = Path(config.checkpoint).parent
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log_path = self.directory / "run.log"
        previous = self.log_path.read_text(encoding="utf-8").splitlines() if self.log_path.exists() else []
        self.log = self.log_path.open("a", encoding="utf-8", buffering=1)
        self.console = Console()
        self.live = None
        self.workers = {}
        self.failures = []
        self.active_step = 0
        for line in previous:
            _, _, message = line.partition(" ")
            self.remember_failure(message)
        self.step = 0
        self.loss = None
        self.memory = {}
        self.attempt = 0
        self.synced = False
        self.status = "Starting"


    def __enter__(self):
        if self.console.is_terminal:
            self.live = Live(self.render(), console=self.console, refresh_per_second=5)
            self.live.start()
        return self


    def __exit__(self, exc_type, exc, tb):
        if exc is not None: self.record(f"FAILED {type(exc).__name__}: {exc}")
        if self.live:
            self.live.update(self.render(), refresh=True)
            self.live.stop()

        files = [self.log_path]
        plots = [
            ("training.png", lambda path: plot_run(self.log_path, path)),
            ("failures.png", lambda path: plot_failures(self.failures, self.config.steps, path)),
            ("predictions.png", lambda path: plot_predictions(self.config, path)),
        ]
        self.log.flush()
        for filename, make_plot in plots:
            path = self.directory / filename
            try:
                result = make_plot(path)
                if result is not False and path.exists():
                    files.append(path)
            except Exception as error:
                self.log.write(f"WARNING could not create {filename}: {error}\n")
                self.console.print(f"Could not create {filename}: {error}", style="yellow")
        self.log.close()
        self.console.print("Files: " + " · ".join(map(str, files)), highlight=False)
        return False


    def remember_failure(self, message):
        """Keep incident details across the whole run, including a resumed run."""
        if not message:
            return
        kind = message.split(maxsplit=1)[0]
        data = fields(message)
        if kind == "STEP":
            self.active_step = int(message.split()[1])
        elif kind == "READY" and self.failures and self.failures[-1]["closed"]:
            failure = self.failures[-1]
            restores = [fields(detail).get("checkpoint") for detail in failure["details"] if detail.startswith("RESTORE ")]
            if restores and data.get("checkpoint") == restores[-1]:
                failure.setdefault("restarted", {})[data["rank"]] = data["pid"]
        elif kind == "FAULT":
            self.failures.append({"step": int(data.get("step", self.active_step)), "details": [message], "closed": False})
        elif kind in {"ERROR", "RECOVERY", "STOPPED", "RESTORE", "FAILED"}:
            if not self.failures or (self.failures[-1]["closed"] and kind != "FAILED"):
                self.failures.append({"step": self.active_step, "details": [], "closed": False})
            self.failures[-1]["details"].append(message)
            if kind in {"RESTORE", "FAILED"}:
                self.failures[-1]["closed"] = True

    def record(self, message):
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        self.log.write(f"{timestamp} {message}\n")
        self.remember_failure(message)
        data = fields(message)
        kind = message.split(maxsplit=1)[0]
        if kind == "GROUP":
            self.attempt = int(data["attempt"])
            self.workers.clear()
            self.synced = False
            self.status = "Starting workers" if self.attempt == 1 else "Restarting workers"
        elif kind == "READY":
            rank = int(data["rank"])
            self.workers[rank] = (data["pid"], data["device"], "online")
            self.step = max(self.step, int(data["checkpoint"]))
            self.status = "Training"
        elif kind == "STEP":
            self.status = f"Running step {message.split()[1]}"
        elif kind == "COMMIT":
            self.step = int(data["step"])
            self.loss = float(data["loss"])
            self.synced = True
            self.status = "Training"
        elif kind == "MEMORY":
            self.memory = data
        elif kind in {"FAULT", "ERROR", "RECOVERY", "FAILED"}:
            self.status = {"FAULT": "Fault injected", "ERROR": "Failure detected", "RECOVERY": "Recovering", "FAILED": "Failed"}[kind]
            if kind == "FAULT" and "rank" in data:
                rank = int(data["rank"])
                if rank in self.workers:
                    pid, device, _ = self.workers[rank]
                    self.workers[rank] = (pid, device, "fault injected")
            if kind == "RECOVERY":
                self.workers.clear()
        elif kind == "COMPLETE":
            self.step = int(data["step"])
            self.status = "Complete"
            self.workers = {rank: (pid, device, "finished") for rank, (pid, device, _) in self.workers.items()}
        if self.live:
            self.live.update(self.render())
        elif kind in {"GROUP", "READY", "FAULT", "ERROR", "RECOVERY", "COMPLETE", "FAILED"}:
            # Pipes, notebooks, and test runners should not receive terminal control codes.
            self.console.print(message, highlight=False, markup=False)

    def render(self):
        settings = f"{self.config.workers} workers  ·  {self.config.model} ({self.config.layers} layers)  ·  {self.config.dataset}  ·  SGD momentum {self.config.momentum:g}"
        step = f"{self.step} / {self.config.steps} committed"
        loss = "—" if self.loss is None else f"{self.loss:.4f}"
        agreement = "replicas agree" if self.synced else "waiting for replica check"
        workers = Table(expand=True, show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
        for title in ("Rank", "Role", "PID", "Device", "Status"):
            workers.add_column(title)
        for rank in range(self.config.workers):
            pid, device, state = self.workers.get(rank, ("—", "—", "starting"))
            workers.add_row(str(rank), "coordinator" if rank == 0 else "worker", str(pid), device, state)
        if self.memory:
            model = kib(self.memory["per_replica_model_bytes"])
            optimizer = kib(self.memory["per_replica_optimizer_bytes"])
            total = kib(self.memory["total_replicated_bytes"])
            memory = f"Per replica: model {model}  ·  optimizer {optimizer}\nAll replicas: {total} (tensor sizes only)"
        else:
            memory = "Waiting for workers…"
        history = []
        for number, failure in enumerate(self.failures, 1):
            details = failure["details"]
            fault = next((item.removeprefix("FAULT ") for item in details if item.startswith("FAULT ")), "unexpected failure")
            fault = re.sub(r" at step=\d+$", "", fault)
            history.append(f"{number}. Step {failure['step']} · {fault}")
            for error in (item for item in details if item.startswith("ERROR ") or item.startswith("FAILED ")):
                history.append(f"    {error}")
            if not any(item.startswith("ERROR ") for item in details):
                history.extend(f"    {item}" for item in details if item.startswith("RECOVERY "))
            stopped = [fields(item) for item in details if item.startswith("STOPPED ")]
            if stopped:
                ranks = ", ".join(f"r{rank['rank']}:pid {rank['pid']}, exit {rank['exit']}" for rank in stopped)
                history.append(f"    Stopped: {ranks}")
            checkpoint = next((fields(item).get("checkpoint") for item in details if item.startswith("RESTORE ")), None)
            if checkpoint is not None:
                restarted = failure.get("restarted", {})
                ready = ", ".join(f"r{rank}:pid {pid}" for rank, pid in sorted(restarted.items())) or "pending"
                history.append(f"    Checkpoint {checkpoint} · restarted: {ready}")
        history_text = "\n".join(history) if history else "No failures so far"
        fault_steps = self.config.fault_steps or (self.config.fault_step,)
        fault_plan = ", ".join(str(step) for step in fault_steps) if self.config.fault != "none" else "none"
        settings += f"  ·  batch {self.config.batch_size}/worker  ·  lr {self.config.lr:g}  ·  timeout {self.config.timeout:g}s"
        status = Text.assemble((self.status, "bold cyan"), f"   |   Step {step}   |   Loss {loss}\n",
                               (f"{settings}\nFault: {self.config.fault}  ·  steps: {fault_plan}  ·  {agreement}", "dim"))
        if self.method == "dp":
            method = ("DP · Centralized data parallelism\n"
                      "Workers process different batches, each with its own model replica.\n"
                      "Rank 0 averages gradients and sends the result to all workers.")
        else:
            method = f"{self.method} · No description available"
        return Panel(Group(status, Panel(method, title="Distributed method", border_style="blue"),
                           Panel(workers, title="Workers", border_style="blue"),
                           Panel(memory, title="Memory", border_style="blue"),
                           Panel(Text(history_text), title=f"Failure history ({len(self.failures)})", border_style="blue")),
                     title="[bold magenta]✦ nano-ddp[/]", subtitle=f"Attempt {self.attempt}  ·  {self.directory}", border_style="magenta")


def plot_run(log_path, output_path):
    """Plot the latest committed value at each step, including across resumed runs."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nano-ddp-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    losses, model_memory, optimizer_memory = {}, {}, {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        _, _, message = line.partition(" ")
        data = fields(message)
        if message.startswith("COMMIT "):
            losses[int(data["step"])] = float(data["loss"])
        elif message.startswith("MEMORY "):
            step = int(data["step"])
            model_memory[step] = int(data["per_replica_model_bytes"]) / 1024
            optimizer_memory[step] = int(data["per_replica_optimizer_bytes"]) / 1024
    if not losses and not model_memory:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    fig.suptitle("nano-ddp · training run", fontsize=14, fontweight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Committed step")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if losses:
        steps = sorted(losses)
        axes[0].plot(steps, [losses[s] for s in steps], color="#a04cc5", marker="o", markersize=3)
    axes[0].set_title("Training loss")
    axes[0].set_ylabel("Cross-entropy loss")
    if model_memory:
        steps = sorted(model_memory)
        axes[1].step(steps, [model_memory[s] for s in steps], where="post", label="Model", color="#4986bf")
        axes[1].step(steps, [optimizer_memory[s] for s in steps], where="post", label="Optimizer", color="#e38a64", linestyle="--")
        axes[1].legend(frameon=False)
    axes[1].set_title("Memory per replica")
    axes[1].set_ylabel("Tensor size (KiB)")
    axes[1].set_ylim(bottom=0)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
