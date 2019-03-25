# TODO batch/layer norm ?
from torch import nn


class FFNN(nn.Module):
    def __init__(self, layers_dim, dropout):
        super(FFNN).__init__()

        layers = []
        for dim_in, dim_out in zip(layers_dim[:-1], layers_dim[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.Dropout(self.dropout))

        self.layer_stack = nn.ModuleList(layers)

    def forward(self, inp):
        h = inp
        for layer in self.layer_stack:
            h = layer(h)
        return h