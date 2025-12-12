import torch
import torch.nn as nn

class BlockCirculantLinear(nn.Module):
    """
    Minimaler Platzhalter fuer spaetere FFT-basierte
    Block-Circulant-Implementierung.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)
