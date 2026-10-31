"""Fault injection for data-parallel training demos."""

import os
from pathlib import Path
import time


def maybe_inject(config, rank, step, events):
    """Inject each configured fault once, even when its step is replayed."""
    target = 0 if config.fault == "coordinator-crash" else 1
    if (
        config.fault == "none"
        or rank != target
        or step not in (config.fault_steps or (config.fault_step,))
    ):
        return

    marker = Path(f"{config.fault_marker}.step-{step}")
    try:
        # The marker survives a restart, so this fault is injected only once.
        with marker.open("x"): pass

    except FileExistsError: return
    
    if config.fault == "slow-worker":
        events.put(f"FAULT rank={rank} sleeping for {config.delay:.1f}s at step={step}")
        time.sleep(config.delay)
    else:
        events.put(f"FAULT rank={rank} crashing at step={step}")
        # Flush the event before deliberately killing the process.
        time.sleep(0.1)
        os._exit(1)

