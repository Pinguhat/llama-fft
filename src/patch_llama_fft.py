import torch
import torch.nn as nn
import os
from typing import Dict
from transformers import AutoModelForCausalLM
from fft_utils import circulant_matvec_fft

# Lokaler Pfad zu deinem Llama-2-7b-hf Modell
MODEL_PATH = "/home/lukas/models/Llama-2-7b-hf"

# Blockgroesse fuer block-circulant Approximation
BLOCK_SIZE = 256
# Wie viele Transformer-Layer patchen wir (erstmal nur 1 fuer Demo)
NUM_LAYERS_TO_PATCH = 1

def _detect_best_convention_for_layer(W: torch.Tensor, block_size: int, *, num_probes: int = 3) -> str:
    """
    Heuristic sanity-check: pick the convention that best matches dense matvec.
    Tests a single top-left block with a few random probe vectors.

    Returns: "diag" or "diag_inv".
    """
    B = block_size
    block = W[:B, :B].to(torch.float32)

    torch.manual_seed(0)
    xs = [torch.randn(B, dtype=torch.float32) for _ in range(num_probes)]

    def score(convention: str) -> float:
        c = dense_block_to_circulant_column(block, convention=convention).to(torch.float32)
        err = 0.0
        for x in xs:
            y_dense = block @ x
            y_circ = circulant_matvec_fft(c, x)
            err += (y_dense - y_circ).pow(2).mean().item()
        return err / float(num_probes)

    e_diag = score("diag")
    e_inv = score("diag_inv")
    return "diag" if e_diag <= e_inv else "diag_inv"

def dense_block_to_circulant_column(W_block: torch.Tensor, *, convention: str = "diag") -> torch.Tensor:
    """
    Least-squares (Frobenius) projection of a dense block onto circulant matrices.

    Convention "diag" (default):
        c[k] = mean_i W_block[i, (i + k) % B]

    Convention "diag_inv" (rarely needed):
        c[k] = mean_i W_block[(i + k) % B, i]
    """
    assert W_block.dim() == 2
    B0, B1 = W_block.shape
    assert B0 == B1
    B = B0
    device = W_block.device
    dtype = W_block.dtype

    if convention not in ("diag", "diag_inv"):
        raise ValueError(f"Unknown convention: {convention}")

    c = torch.zeros(B, device=device, dtype=dtype)
    idx = torch.arange(B, device=device)

    for k in range(B):
        if convention == "diag":
            cols = (idx + k) % B
            vals = W_block[idx, cols]
        else:
            rows = (idx + k) % B
            vals = W_block[rows, idx]
        c[k] = vals.mean()

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
            convention = _detect_best_convention_for_layer(W, block_size)
            print(f"      Using circulant convention: {convention}")


            for j in range(out_blocks):
                for i in range(in_blocks):
                    block = W_blocks[j, i]   # (B, B)
                    c_ji = dense_block_to_circulant_column(block, convention=convention)
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

def _get_submodule_by_name(model: nn.Module, name: str) -> nn.Module:
    cur = model
    if name == "":
        return cur
    for part in name.split("."):
        cur = getattr(cur, part)
    return cur


def save_bc_params(model: nn.Module, path: str) -> None:
    """
    Save only BlockCirculantLinear parameters (c and bias) to a compact checkpoint.
    """
    bc_state: Dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, BlockCirculantLinear):
            bc_state[f"{module_name}.c"] = module.c.detach().cpu()
            if module.bias is not None:
                bc_state[f"{module_name}.bias"] = module.bias.detach().cpu()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(bc_state, path)
    print(f"Saved BC params: {len(bc_state)} tensors -> {path}")


def load_bc_params(model: nn.Module, path: str, *, strict_shapes: bool = True) -> None:
    """
    Load BlockCirculantLinear parameters (c and bias) from checkpoint created by save_bc_params().
    """
    state: Dict[str, torch.Tensor] = torch.load(path, map_location="cpu")
    loaded = 0
    skipped = 0

    for full_name, tensor in state.items():
        if not (full_name.endswith(".c") or full_name.endswith(".bias")):
            skipped += 1
            continue

        module_name, param_name = full_name.rsplit(".", 1)
        try:
            module = _get_submodule_by_name(model, module_name)
        except AttributeError:
            skipped += 1
            continue

        if not hasattr(module, param_name):
            skipped += 1
            continue

        param = getattr(module, param_name)
        if not isinstance(param, torch.Tensor):
            skipped += 1
            continue

        if strict_shapes and param.shape != tensor.shape:
            skipped += 1
            continue

        with torch.no_grad():
            param.copy_(tensor.to(param.device, dtype=param.dtype))
        loaded += 1

    print(f"Loaded BC params from {path}: loaded={loaded}, skipped={skipped}")


def patch_mlp_with_block_circulant(
    model: AutoModelForCausalLM,
    *,
    num_layers_to_patch: int = NUM_LAYERS_TO_PATCH,
    block_size: int = BLOCK_SIZE
) -> None:
    """
    Replace MLP linear layers in the first NUM_LAYERS_TO_PATCH transformer
    layers with BlockCirculantLinear layers that approximate the original
    dense weights.
    """
    n_layers = model.config.num_hidden_layers
    print(f"Model has {n_layers} transformer layers.")
    print(f"Patching the first {num_layers_to_patch} layer(s).")

    for layer_idx in range(num_layers_to_patch):
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

            new_layer = BlockCirculantLinear.from_linear(
                old_layer,
                block_size=block_size,
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
