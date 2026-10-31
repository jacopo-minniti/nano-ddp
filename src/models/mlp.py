from torch import nn


class MLP(nn.Module):
    """A configurable multilayer perceptron for classification."""

    def __init__(self, num_layers=2, input_size=28 * 28, hidden_size=64, output_size=10):
        super().__init__()
        if num_layers < 1: raise ValueError("the MLP must have at least one layer")

        layers = [nn.Flatten()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU()])
            input_size = hidden_size

        layers.append(nn.Linear(input_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, images): return self.layers(images)
