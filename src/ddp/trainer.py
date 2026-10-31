"""Future decentralized data parallel training."""

from src.distributed import ParallelTrainer


class NanoDDP(ParallelTrainer):
    """Reserved for training with decentralized gradient synchronization."""

    def run(self):
        raise NotImplementedError("NanoDDP has not been implemented")
