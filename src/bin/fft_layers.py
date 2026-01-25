import torch
import torch.nn as nn
from .fft_utils import circulant_matvec_fft

class CirculantLinear(nn.Module):
    """
    Simple circulant linear layer.

    Each output dimension i uses a circulant matrix C_i,
    represented by its first column c_i of length n = in_features.

    Forward pass:
    y_i = C_i x  for each output dimension i,
    computed via FFT.

    This is mainly a demo for FFT based matrix vector products.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters: first column of each circulant matrix
        # Shape: (out_features, in_features)
        self.c = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch, in_features)
        returns: shape (batch, out_features)
        """
        assert x.dim() == 2
        assert x.shape[1] == self.in_features

        batch_size = x.shape[0]
        device = x.device

        # Output tensor
        y = torch.empty(batch_size, self.out_features, device=device, dtype=x.dtype)

        # Compute each output dimension separately via FFT based circulant matvec
        for i in range(self.out_features):
            c_i = self.c[i]  # shape (in_features,)
            # Apply C_i to all batch vectors
            # We do this in a loop over batch for clarity.
            # (Could be vectorized, but this is easier to understand.)
            for b in range(batch_size):
                y[b, i] = circulant_matvec_fft(c_i, x[b])

        if self.bias is not None:
            y = y + self.bias

        return y
