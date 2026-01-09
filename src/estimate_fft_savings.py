import math

# Llama-2-7b MLP-Dimensionen
D_MODEL = 4096      # hidden size
D_FF = 11008        # intermediate size
BLOCK_SIZE = 256    # block size fuer block-circulant

def dense_linear_cost(in_f: int, out_f: int) -> int:
    """
    Approximate number of MACs (multiplications ~ additions)
    fuer einen dichten Linear-Layer: y = x W^T
    """
    macs = in_f * out_f
    return macs
 
def fft_circulant_block_cost(B: int):
    """
    Schaetzung der Kosten fuer EINE circulant-matvec mit FFT
    der Laenge B.

    Modell (grob):
      - eine FFT (rfft oder irfft) kostet ~ 5 * B * log2(B) multiplications
        und etwa genauso viele additions
      - wir machen: 2x rfft + 1x irfft + B complex multiplies
        (1 complex multiply ~ 4 real mult + 2 real add)

    Rueckgabe:
      (real_multiplications, real_additions) pro circulant-matvec
    """
    log2B = math.log2(B)

    fft_mult = 5 * B * log2B
    fft_add = 5 * B * log2B

    mult = 3 * fft_mult + 4 * B   # 3 FFTs + B complex multiplies
    add = 3 * fft_add + 2 * B     # 3 FFTs + B complex multiplies

    return mult, add

def block_circulant_linear_cost(in_f: int, out_f: int, B: int):
    """
    Kosten fuer einen BlockCirculantLinear:
      - Matrix W in BxB-Bloecke geteilt
      - fuer jeden Block eine circulant matvec per FFT
    """
    assert in_f % B == 0 and out_f % B == 0
    in_blocks = in_f // B
    out_blocks = out_f // B
    blocks = in_blocks * out_blocks

    m_block, a_block = fft_circulant_block_cost(B)
    mult = blocks * m_block
    add = blocks * a_block
    return mult, add

def main():
    d_model = D_MODEL
    d_ff = D_FF
    B = BLOCK_SIZE

    print(f"d_model = {d_model}, d_ff = {d_ff}, block_size = {B}")

    # 1) Dichte Linear-Kosten pro Token (4096 -> 11008)
    dense_macs = dense_linear_cost(d_model, d_ff)
    print(f"Dense linear (4096 -> 11008): {dense_macs:.0f} MACs pro Token")

    # 2) Block-circulant-FFT fuer dieselbe Shape
    fft_mult, fft_add = block_circulant_linear_cost(d_model, d_ff, B)
    print("\nBlock-circulant mit FFT (4096 -> 11008):")
    print(f"  ~{fft_mult:.0f} Multiplikationen pro Token")
    print(f"  ~{fft_add:.0f} Additionen pro Token")

    # 3) Vergleich pro Linear-Layer
    print("\nVergleich pro Linear-Layer:")
    print(f"  Dense mult : {dense_macs:.0f}")
    print(f"  FFT  mult  : {fft_mult:.0f}")
    print(f"  Speedup (mult) ~ {dense_macs / fft_mult:.2f}x")

    # 4) Ganzer MLP-Block (gate_proj, up_proj, down_proj)
    #    MLP hat 3 Linears:
    #      2x (d_model -> d_ff)  und  1x (d_ff -> d_model)
    dense_mlp_macs = 3 * dense_macs

    fft_mult_up, fft_add_up = fft_mult, fft_add
    fft_mult_down, fft_add_down = block_circulant_linear_cost(d_ff, d_model, B)
    fft_mlp_mult = 2 * fft_mult_up + fft_mult_down
    fft_mlp_add = 2 * fft_add_up + fft_add_down

    print("\nPro MLP-Block (gate, up, down):")
    print(f"  Dense MACs (approx) : {dense_mlp_macs:.0f}")
    print(f"  FFT mult  (approx)  : {fft_mlp_mult:.0f}")
    print(f"  FFT add   (approx)  : {fft_mlp_add:.0f}")
    print(f"  Speedup (mult) ~ {dense_mlp_macs / fft_mlp_mult:.2f}x")

if __name__ == '__main__':
    main()
