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
    # Use float32 for FFT internally 
    orig_dtype = x.dtype
    c32 = c.to(torch.float32)
    x32 = x.to(torch.float32)

    fft_c = torch.fft.rfft(c32)
    fft_x = torch.fft.rfft(x32)

    # Elementwise complex multiply in frequency domain
    fft_y = fft_c * fft_x

    # Back to time domain
    y32 = torch.fft.irfft(fft_y, n=n)

    return y32.to(orig_dtype)
