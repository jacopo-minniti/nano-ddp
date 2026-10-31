# nano-ddp

A small distributed-training experiment for a distributed systems class. Independent Python processes train replicas of an MLP on MNIST. Rank 0 averages the gradients and sends the result back to the other workers. A separate supervisor detects failures and restarts the group from its last checkpoint.

This repository implements centralized data parallelism. Ring AllReduce and DDP are placeholders for future work.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run:

```sh
uv sync
uv run nano-ddp --run-dir runs/normal
```

The first run downloads MNIST to `data/`. Python 3.12–3.14 is required. By default each worker selects CUDA if available, then Apple MPS, then CPU. Use `--device cpu` to force CPU. Processes are real, separate operating-system processes communicating over local pipes.

## Training and recovery

The default choices are `--model mlp --dataset mnist`. The MLP has two linear layers by default, with ReLU between hidden layers. Set `--layers` to choose the total number of linear layers (including the output layer); `--layers 1` is a linear classifier.

Each step uses disjoint, equal-sized parts of a deterministic global MNIST batch. Rank 0 averages the gradients, all replicas apply the same SGD update, and rank 0 checks that their model hashes match. It also checks that optimizer states match, then writes the model and optimizer state to a checkpoint using an atomic rename. SGD momentum is optional and defaults to zero.

If a process crashes or a worker fails to respond before the timeout, the supervisor stops the entire group and starts it again from the last checkpoint. An unfinished step is replayed. The terminal dashboard shows worker status, the latest committed step, loss, memory, and recovery events. The run directory contains `run.log` with the full event history and `training.png` with training loss and per-replica model and optimizer tensor sizes. `failures.png` shows faults and available recovery checkpoints; `predictions.png` compares MNIST test examples with their predicted and true labels. Resumed runs append to the log and update the plot. The supervisor itself is a single point of failure. Checkpoints are local files; this is not recovery from machine or disk failure. Timeouts can mistake a slow worker for a failed one.

```sh
# Kill a non-coordinator worker at step 2.
uv run nano-ddp --device cpu --run-dir runs/worker --fault worker-crash

# Kill rank 0, the gradient coordinator, at step 2.
uv run nano-ddp --device cpu --run-dir runs/coordinator --fault coordinator-crash

# Delay a worker longer than the coordinator's timeout.
uv run nano-ddp --device cpu --run-dir runs/slow --fault slow-worker --timeout 4 --delay 8
```

Schedule the same fault more than once by passing several steps. Each scheduled fault happens once, including across restarts and resumed runs:

```sh
uv run nano-ddp --device cpu --steps 100 --fault worker-crash \
  --fault-step 20 40 60 --run-dir runs/multiple-faults
```

`--fault-step 20 --fault-step 40` works too. By default, the restart limit accommodates the scheduled faults; use `--max-restarts` to set a different limit. The dashboard keeps a failure history with the affected rank, detected errors, each stopped process and the checkpoint available for recovery. The full history is also in `run.log`, including across resumed runs.

Use a new `--run-dir` for each demonstration. To resume a previous run, pass its `--run-dir` and `--resume` with the same worker count, batch size, layer count, learning rate, momentum and device; set `--steps` to the desired total number of steps. Per-step fault markers in the run directory prevent a fault from repeating on resume. Remove the run directory to start that demo again.

`--workers` includes rank 0; `--batch-size` is per worker. Use `--help` for all options. Increase `--timeout` for slow machines. On a shared accelerator, several workers share that device; this project does not measure realistic multi-GPU performance.

## Extending the trainer

Application code is in `src/` and tests are in `tests/`. The Python package is `src`, matching the source directory.

Model constructors are registered in `src/models/__init__.py`; each registered constructor accepts `num_layers`. Dataset factories are registered in `src/data.py` and receive the data directory and a download flag. A dataset must support indexing and return `(input_tensor, class_label)` samples. Add a constructor or factory to its registry to make it selectable with `--model` or `--dataset`.

The current training loop expects a classification model and uses cross-entropy loss. A language model or another training task will also need its own batching and loss logic. New checkpoints record the model, dataset, layer count, worker count and batch size to detect incompatible resumes; older checkpoints do not contain this metadata.

`distributed.py` holds the shared trainer interface and helpers. `dp/` contains the working centralized trainer; `ddp/` holds the future decentralized trainer and the `AllReduce` interface. `RingAllReduce`, `NanoDDP`, and `NanoGPT2` are unimplemented placeholders that raise `NotImplementedError`. The current centralized trainer does not use them.

## SGD momentum and memory

Use `--momentum 0.9` to enable momentum. Each worker keeps its own momentum buffers, which are restored with the model after a failure. Use the same `--lr` and `--momentum` when resuming a run.

```sh
uv run nano-ddp --device cpu --run-dir runs/momentum --layers 3 --momentum 0.9 --fault worker-crash
```

The coordinator logs model and optimizer tensor bytes per replica, their combined size across all replicas, and checkpoint file size. With momentum, optimizer buffers start at zero bytes and are allocated after the first update.