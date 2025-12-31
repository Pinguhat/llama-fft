import torch

def circulant_matvec_fft(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Multiplies a circulant matrix C (given by first column c)
    with a vector x using FFT.

    Both c and x have shape (n,).
    Returns y with shape (n,), where y = C x.

    Complexity: O(n log n) via FFT instead of O(n^2) for dense matmul.
    """
    # Ensure 1D
    assert c.dim() == 1
    assert x.dim() == 1
    n = c.shape[0]
    assert x.shape[0] == n

    # FFT (real to complex)
    fft_c = torch.fft.rfft(c)
    fft_x = torch.fft.rfft(x)

    # Elementwise complex multiply in frequency domain
    fft_y = fft_c * fft_x

    # Back to time domain
    y = torch.fft.irfft(fft_y, n=n)

    return y
