"""Interfaces for future decentralized gradient collectives."""

from abc import ABC, abstractmethod


class AllReduce(ABC):
    """Collectively average gradients and return the result on every rank.

    Participants must call the collective in the same order. Failure, timeout,
    buffer ownership, and process-group semantics are left for implementation.
    """

    @abstractmethod
    def all_reduce(self, gradients):
        """Return averaged gradient buffers for all participating ranks."""
        raise NotImplementedError


class RingAllReduce(AllReduce):
    """Placeholder for a ring-based collective; no transport exists yet."""

    def __init__(self):
        raise NotImplementedError("Ring AllReduce has not been implemented")

    def all_reduce(self, gradients):
        raise NotImplementedError("Ring AllReduce has not been implemented")
