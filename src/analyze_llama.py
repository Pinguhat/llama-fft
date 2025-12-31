import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# Local path to your downloaded Llama-2-7b-hf
MODEL_NAME = "/home/lukas/models/Llama-2-7b-hf"


def analyze_linear_layers(model):
    total_layers = 0
    total_params = 0
    total_macs_per_token = 0

    print("Listing all nn.Linear layers in the model:\n")
    print(f"{'#':>4s} | {'name':60s} | {'in':>6s} -> {'out':>6s} | {'params':>12s} | {'MACs/token':>12s}")
    print("-" * 110)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_layers += 1
            in_f = module.in_features
            out_f = module.out_features

            weight_params = in_f * out_f
            bias_params = out_f if module.bias is not None else 0
            params = weight_params + bias_params

            macs_per_token = in_f * out_f

            total_params += params
            total_macs_per_token += macs_per_token

            print(
                f"{total_layers:4d} | {name:60s} | "
                f"{in_f:6d} -> {out_f:6d} | "
                f"{params:12d} | {macs_per_token:12d}"
            )

    print("\nSummary (from real model):")
    print(f"Total nn.Linear layers:           {total_layers}")
    print(f"Total parameters in nn.Linear:    {total_params:,}")
    print(f"Total MACs per token (all Lin.):  {total_macs_per_token:,}")
    print(f"Approx FLOPs per token (Lin.):    {2 * total_macs_per_token:,}")
    print("  (FLOPs ~= 2 * MACs: mul + add)")


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.float16,  # if this breaks, try torch.float32
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    model.eval()
    analyze_linear_layers(model)


if __name__ == "__main__":
    main()
