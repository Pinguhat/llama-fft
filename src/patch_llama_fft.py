import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# Lokaler Pfad zu deinem Llama-2-7b-hf Modell
MODEL_PATH = "/home/lukas/models/Llama-2-7b-hf"

# Blockgroesse fuer block-circulant Approximation
BLOCK_SIZE = 256
# Wie viele Transformer-Layer patchen wir (erstmal nur 1 fuer Demo)
NUM_LAYERS_TO_PATCH = 1


def circulant_matvec_fft(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Multiply a circulant matrix C (given by its first column c)
    with a vector x using FFT.

    c: shape (n,)
    x: shape (n,)
    returns y: shape (n,), where y = C x

    Uses float32 for FFT internally for stability and converts back.
    """
    assert c.dim() == 1
    assert x.dim() == 1
    n = c.shape[0]
    assert x.shape[0] == n

    orig_dtype = x.dtype

    c32 = c.to(torch.float32)
    x32 = x.to(torch.float32)

    fft_c = torch.fft.rfft(c32)
    fft_x = torch.fft.rfft(x32)

    fft_y = fft_c * fft_x

    y32 = torch.fft.irfft(fft_y, n=n)

    return y32.to(orig_dtype)


def dense_block_to_circulant_column(W_block: torch.Tensor) -> torch.Tensor:
    """
    Approximate a dense BxB weight block W_block by a circulant matrix C
    and return the first column c of C.

    Simple heuristic:
        c[k] = mean_i W_block[i, (i + k) % B]
    That means: c[k] is the mean of the cyclic diagonal with offset k.
    """
    assert W_block.dim() == 2
    B0, B1 = W_block.shape
    assert B0 == B1
    B = B0
    device = W_block.device
    dtype = W_block.dtype

    c = torch.zeros(B, device=device, dtype=dtype)
    idx = torch.arange(B, device=device)

    for k in range(B):
        cols = (idx + k) % B
        diag_vals = W_block[idx, cols]
        c[k] = diag_vals.mean()

    return c


class BlockCirculantLinear(nn.Module):
    """
    Block-circulant linear layer with FFT based multiplication.

    Weight matrix W has shape (out_features, in_features) and is represented
    as a grid of circulant blocks of size (block_size x block_size).

    in_features  = in_blocks  * block_size
    out_features = out_blocks * block_size

    Parameters:
        c[j, i, :] is the first column of the circulant block C_{j,i}
        that maps input block i to output block j.
    """

    def __init__(self, in_features: int, out_features: int, block_size: int = 256, bias: bool = True):
        super().__init__()
        assert in_features % block_size == 0, "in_features must be divisible by block_size"
        assert out_features % block_size == 0, "out_features must be divisible by block_size"

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.in_blocks = in_features // block_size
        self.out_blocks = out_features // block_size

        # Parameters: first column of each block C_{j,i}
        # Shape: (out_blocks, in_blocks, block_size)
        self.c = nn.Parameter(
            torch.randn(self.out_blocks, self.in_blocks, self.block_size) * 0.01
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_size: int = 256) -> "BlockCirculantLinear":
        """
        Create a BlockCirculantLinear layer that approximates
        an existing dense nn.Linear layer.

        The dense weight matrix is partitioned into BxB blocks and each
        block is approximated by a circulant matrix (via dense_block_to_circulant_column).
        """
        in_f = linear.in_features
        out_f = linear.out_features

        layer = cls(
            in_features=in_f,
            out_features=out_f,
            block_size=block_size,
            bias=(linear.bias is not None),
        )

        with torch.no_grad():
            W = linear.weight.data.clone()  # (out_f, in_f)
            B = block_size
            out_blocks = layer.out_blocks
            in_blocks = layer.in_blocks

            # Reshape W into blocks: (out_blocks, in_blocks, B, B)
            # Start from (out_f, in_f) = (out_blocks * B, in_blocks * B)
            W_blocks = W.view(out_blocks, B, in_blocks, B)  # (out_blocks, B, in_blocks, B)
            W_blocks = W_blocks.permute(0, 2, 1, 3)         # (out_blocks, in_blocks, B, B)

            for j in range(out_blocks):
                for i in range(in_blocks):
                    block = W_blocks[j, i]   # (B, B)
                    c_ji = dense_block_to_circulant_column(block)
                    layer.c.data[j, i].copy_(c_ji)

            if layer.bias is not None and linear.bias is not None:
                layer.bias.data.copy_(linear.bias.data)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unterstuetzt sowohl
          x: (batch, in_features)        -> (batch, out_features)
        als auch
          x: (batch, seq_len, in_features) -> (batch, seq_len, out_features)
        """
        if x.dim() == 3:
            # (batch, seq, in_features) -> zu (batch * seq, in_features) flatten
            batch_size, seq_len, in_f = x.shape
            assert in_f == self.in_features
            x_flat = x.reshape(batch_size * seq_len, in_f)
            is_3d = True
        elif x.dim() == 2:
            batch_size, in_f = x.shape
            assert in_f == self.in_features
            x_flat = x
            seq_len = 1
            is_3d = False
        else:
            raise ValueError(f"Unsupported input dim: {x.dim()} (expected 2 or 3)")

        device = x_flat.device
        dtype = x_flat.dtype

        # Reshape input into blocks: (batch_flat, in_blocks, block_size)
        x_blocks = x_flat.view(x_flat.shape[0], self.in_blocks, self.block_size)

        # Output blocks: (batch_flat, out_blocks, block_size)
        y_blocks = torch.zeros(
            x_flat.shape[0], self.out_blocks, self.block_size, device=device, dtype=dtype
        )

        # Explizite Schleifen fuer Verstaendlichkeit (nicht optimiert):
        for b in range(x_flat.shape[0]):
            for j in range(self.out_blocks):
                acc = torch.zeros(self.block_size, device=device, dtype=dtype)
                for i in range(self.in_blocks):
                    c_ji = self.c[j, i]       # (block_size,)
                    x_bi = x_blocks[b, i]     # (block_size,)
                    acc = acc + circulant_matvec_fft(c_ji, x_bi)
                y_blocks[b, j] = acc

        # Merge blocks back: (batch_flat, out_features)
        y_flat = y_blocks.view(x_flat.shape[0], self.out_features)

        if self.bias is not None:
            y_flat = y_flat + self.bias

        if is_3d:
            # Zurueck zu (batch, seq_len, out_features)
            y = y_flat.view(batch_size, seq_len, self.out_features)
        else:
            y = y_flat  # (batch, out_features)

        return y



def patch_mlp_with_block_circulant(model: AutoModelForCausalLM) -> None:
    """
    Replace MLP linear layers in the first NUM_LAYERS_TO_PATCH transformer
    layers with BlockCirculantLinear layers that approximate the original
    dense weights.
    """
    n_layers = model.config.num_hidden_layers
    print(f"Model has {n_layers} transformer layers.")
    print(f"Patching the first {NUM_LAYERS_TO_PATCH} layer(s).")

    for layer_idx in range(NUM_LAYERS_TO_PATCH):
        mlp = model.model.layers[layer_idx].mlp
        print(f"  Patching layer {layer_idx} MLP...")

        for name in ["gate_proj", "up_proj", "down_proj"]:
            old_layer = getattr(mlp, name)
            if not isinstance(old_layer, nn.Linear):
                print(f"    Skip {name}: not an nn.Linear")
                continue

            print(
                f"    Replacing {name}: "
                f"in_features={old_layer.in_features}, out_features={old_layer.out_features}"
            )

            # Initialize new block-circulant layer from original dense weights
            new_layer = BlockCirculantLinear.from_linear(
                old_layer,
                block_size=BLOCK_SIZE,
            )
            setattr(mlp, name, new_layer)


def main():
    print(f"Loading Llama-2-7b from local path: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()

    # Patch model in-place
    patch_mlp_with_block_circulant(model)

    # Optional: small forward pass to check shapes
    print("Running a small forward pass to check shapes...")
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 8))  # batch=1, seq_len=8

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    print("Forward pass completed.")
    print("Logits shape:", outputs.logits.shape)


if __name__ == "__main__":
    main()
