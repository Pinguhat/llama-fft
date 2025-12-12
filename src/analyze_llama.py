import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"

def analyze_linear_layers(model):
    """
    Geht durch alle Module des Llama Modells, findet nn.Linear,
    zaehlt sie und berechnet Parameterzahl und MACs pro Token.
    """
    total_layers = 0
    total_params = 0
    total_macs_per_token = 0

    print("Listing all nn.Linear layers in Llama 3.1:\n")
    print(f"{'#':>4s} | {'name':70s} | {'in':>6s} -> {'out':>6s} | {'params':>12s} | {'MACs/token':>12s}")
    print("-" * 130)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_layers += 1
            in_f = module.in_features
            out_f = module.out_features

            weight_params = in_f * out_f
            bias_params = out_f if module.bias is not None else 0
            params = weight_params + bias_params

            # MACs pro Token fuer einen Linear-Layer:
            # y = x * W^T + b  -> in_f * out_f MACs
            macs_per_token = in_f * out_f

            total_params += params
            total_macs_per_token += macs_per_token

            print(
                f"{total_layers:4d} | {name:70s} | "
                f"{in_f:6d} -> {out_f:6d} | "
                f"{params:12d} | {macs_per_token:12d}"
            )

    print("\nSummary for all nn.Linear layers in Llama 3.1:")
    print(f"Total nn.Linear layers:           {total_layers}")
    print(f"Total parameters in nn.Linear:    {total_params:,}")
    print(f"Total MACs per token (all Lin.):  {total_macs_per_token:,}")
    print("Approx FLOPs per token (Lin.):    ", 2 * total_macs_per_token)
    print("Note: FLOPs ~= 2 * MACs (mul + add).")


def main():
    print(f"Loading model: {MODEL_NAME}")
    # Achtung: Llama 3.1 ist gross, du brauchst genug RAM / VRAM.
    # Wenn du nur analysieren willst, kannst du auch device_map='cpu' nutzen.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",  # oder "auto", wenn du GPU / accelerate nutzt
        torch_dtype=torch.float16  # oder float32, wenn CPU-RAM reicht
    )
    model.eval()

    analyze_linear_layers(model)


if __name__ == "__main__":
    main()
