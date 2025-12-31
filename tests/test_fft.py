import torch
from src.fft_layers import CirculantLinear

def test_circulant_linear_shapes():
    layer = CirculantLinear(in_features=16, out_features=8)
    x = torch.randn(4, 16)  # batch=4
    y = layer(x)
    assert y.shape == (4, 8)
