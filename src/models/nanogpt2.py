"""Reserved for a future language model; not supported by the classifier yet."""

from torch import nn


class NanoGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("NanoGPT2 has not been implemented")
