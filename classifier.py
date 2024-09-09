import math
import torch
import torch.nn as nn


class TrafficClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers_dim: list = None):
        if hidden_layers_dim is None:
            hidden_layers_dim = [(2 ** math.ceil(math.log2(input_dim))) // 2,
                                 (2 ** math.ceil(math.log2(input_dim))) // 4,
                                 (2 ** math.ceil(math.log2(input_dim))) // 8]
        super(TrafficClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self._out_dim = output_dim
        self.layers.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_layers_dim[0]),
                nn.ReLU()
            )
        )
        for i, layer in enumerate(hidden_layers_dim):
            if i + 1 < len(hidden_layers_dim):
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_layers_dim[i], hidden_layers_dim[i + 1]),
                        nn.ReLU()
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_layers_dim[i], output_dim),
                        nn.Sigmoid() if output_dim == 1 else nn.Softmax(dim=1)
                    )
                )
                break

    def forward(self, x):
        x = x.view(x.size()[0], -1)  # flattening the tensor (necessary for fully connected layers)
        for layer in self.layers:
            x = layer(x)
        if self._out_dim == 1:
            x = x.squeeze()
        return x
