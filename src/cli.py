import argparse
from pathlib import Path

from src.data import DATASETS
from src.distributed import Config
from src.dp import DataParallelTrainer
from src.models import MODEL_TYPES


def main():
    parser = argparse.ArgumentParser(description="Train a classifier with centralized data parallelism.")

    parser.add_argument("--model", choices=MODEL_TYPES, default="mlp")
    parser.add_argument("--dataset", choices=DATASETS, default="mnist")
    parser.add_argument("--workers", type=int, default=3, help="total workers, including rank 0 (default: 3)")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--layers", type=int, default=2, help="number of MLP linear layers (default: 2)")
    parser.add_argument("--batch-size", type=int, default=32, help="per-worker batch size")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.0, help="SGD momentum in [0, 1); default: 0")
    parser.add_argument("--timeout", type=float, default=10, help="seconds to wait for each worker message")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--run-dir", default="runs/default")
    parser.add_argument("--resume", action="store_true", help="resume the checkpoint in --run-dir")
    parser.add_argument("--max-restarts", type=int, default=None, help="restart limit (default: at least 2 and enough for the scheduled faults)")
    parser.add_argument("--fault", choices=("none", "worker-crash", "coordinator-crash", "slow-worker"), default="none")
    parser.add_argument("--fault-step", type=int, nargs="+", action="append", help="one or more steps; may be specified more than once (default: 2)")
    parser.add_argument("--delay", type=float, default=20, help="seconds to sleep for slow-worker injection")

    args = parser.parse_args()
    run = Path(args.run_dir)
    checkpoint = run / "checkpoint.pt"

    if checkpoint.exists() and not args.resume:
        parser.error("checkpoint already exists; choose a new --run-dir or use --resume")
    if args.resume and not checkpoint.exists():
        parser.error("no checkpoint exists in --run-dir")
    if args.delay < 0:
        parser.error("delay cannot be negative")

    fault_steps = sorted({step for group in (args.fault_step or [[2]]) for step in group})
    config = Config(
        data_dir=args.data_dir, 
        checkpoint=str(checkpoint), 
        workers=args.workers, 
        model=args.model, 
        dataset=args.dataset, 
        layers=args.layers,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        timeout=args.timeout,
        device=args.device,
        max_restarts=args.max_restarts,
        fault=args.fault,
        fault_step=fault_steps[0],
        fault_steps=tuple(fault_steps),
        delay=args.delay
    )
    
    DataParallelTrainer(config).run()


if __name__ == "__main__": main()
