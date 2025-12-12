import torch
from src.fft_layers import BlockCirculantLinear

def test_forward():
    layer = BlockCirculantLinear(16, 8)
    x = torch.randn(2, 16)
    y = layer(x)
    assert y.shape == (2, 8)
