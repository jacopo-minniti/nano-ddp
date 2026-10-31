"""Small end-of-run visualizations for the current classifier and failures."""

import os
import re
import tempfile
from pathlib import Path


def pyplot():
    # Matplotlib may otherwise try to write configuration outside the run environment.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nano-ddp-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_predictions(config, output_path):
    """Show deterministic MNIST test samples using the last committed model."""
    if config.dataset != "mnist" or not Path(config.checkpoint).exists():
        return False

    import torch
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    from src.models import build_model
    from src.training import load_checkpoint

    model = build_model(config.model, torch.device("cpu"), config.layers)
    step = load_checkpoint(config.checkpoint, model)
    if step == 0:
        return False
    model.eval()
    dataset = MNIST(config.data_dir, train=False, transform=ToTensor(), download=False)
    samples = [dataset[index] for index in range(12)]
    images = torch.stack([image for image, _ in samples])
    with torch.no_grad():
        predictions = model(images).argmax(dim=1).tolist()

    plt = pyplot()
    fig, axes = plt.subplots(3, 4, figsize=(8, 6), layout="constrained")
    fig.suptitle(f"MNIST test examples · checkpoint step {step}", fontsize=14, fontweight="bold")
    for axis, (image, label), prediction in zip(axes.flat, samples, predictions, strict=True):
        axis.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        color = "#267955" if prediction == label else "#ba4a50"
        axis.set_title(f"Predicted {prediction}  ·  True {label}", color=color, fontsize=10)
        axis.axis("off")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def plot_failures(failures, total_steps, output_path):
    """Show fault steps and the checkpoints available for recovery, not elapsed time."""
    plt = pyplot()
    from matplotlib.ticker import MaxNLocator

    height = max(3.0, min(12.0, 1.8 + len(failures) * 0.6))
    fig, axis = plt.subplots(figsize=(9, height), layout="constrained")
    axis.set_title("Faults and recovery checkpoints", fontsize=14, fontweight="bold")
    axis.set_xlabel("Training step (not elapsed time)")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=0.2)
    last_step = max([1, total_steps] + [incident["step"] for incident in failures])
    axis.set_xlim(-0.5, last_step + 0.5)

    if not failures:
        axis.set_yticks([])
        axis.text(0.5, 0.5, "No fault or failure events recorded", transform=axis.transAxes,
                  ha="center", va="center", color="#666666")
    else:
        labels = []
        for number, incident in enumerate(failures, 1):
            step = incident["step"]
            details = incident["details"]
            fault = next((item.removeprefix("FAULT ") for item in details if item.startswith("FAULT ")), "failure detected")
            fault = re.sub(r" at step=\d+$", "", fault)
            labels.append(f"#{number} · {fault}")
            checkpoint = next((int(match.group(1)) for item in details
                               if (match := re.search(r"^RESTORE checkpoint=(\d+)", item))), None)
            axis.scatter(step, number, color="#c65c63", s=55, zorder=3,
                         label="Fault / failure" if number == 1 else None)
            if checkpoint is not None:
                axis.plot([checkpoint, step], [number, number], color="#a7a7a7", linewidth=1.5, zorder=1)
                axis.scatter(checkpoint, number, marker="s", color="#4a85b3", s=45, zorder=3,
                             label="Checkpoint available" if not any(
                                 "RESTORE " in d for earlier in failures[:number - 1] for d in earlier["details"]
                             ) else None)
        axis.set_yticks(range(1, len(labels) + 1), labels)
        axis.invert_yaxis()
        axis.legend(loc="lower right", frameon=False)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True
