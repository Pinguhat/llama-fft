from transformers import AutoConfig

MODEL_NAME = "meta-llama/Llama-2-7b-hf"


def analyze_from_config(cfg):
    n_layer = cfg.num_hidden_layers
    d_model = cfg.hidden_size
    d_ff = cfg.intermediate_size
    n_heads = cfg.num_attention_heads
    vocab_size = cfg.vocab_size

    print("Config:")
    print(f"  num_hidden_layers  : {n_layer}")
    print(f"  hidden_size (d_model): {d_model}")
    print(f"  intermediate_size (d_ff): {d_ff}")
    print(f"  num_attention_heads: {n_heads}")
    print(f"  vocab_size         : {vocab_size}")
    print()

    def params_linear(m_in: int, m_out: int, bias: bool = True) -> int:
        return m_in * m_out + (m_out if bias else 0)

    def macs_linear(m_in: int, m_out: int) -> int:
        return m_in * m_out  # MACs pro Token

    layer_index = 0
    total_params = 0
    total_macs_per_token = 0
    total_linear_layers = 0

    print("Per-block linear layers (theoretical Llama-2 style):\n")
    print(f"{'#':>4s} | {'name':35s} | {'in':>6s} -> {'out':>6s} | {'params':>12s} | {'MACs/token':>12s}")
    print("-" * 90)

    for block in range(n_layer):
        # Attention
        for proj_name, m_in, m_out in [
            (f"block{block}.attn.q_proj", d_model, d_model),
            (f"block{block}.attn.k_proj", d_model, d_model),
            (f"block{block}.attn.v_proj", d_model, d_model),
            (f"block{block}.attn.o_proj", d_model, d_model),
        ]:
            layer_index += 1
            total_linear_layers += 1
            p = params_linear(m_in, m_out, bias=True)
            m = macs_linear(m_in, m_out)
            total_params += p
            total_macs_per_token += m
            print(f"{layer_index:4d} | {proj_name:35s} | {m_in:6d} -> {m_out:6d} | {p:12d} | {m:12d}")

        # MLP
        for proj_name, m_in, m_out in [
            (f"block{block}.mlp.gate_proj", d_model, d_ff),
            (f"block{block}.mlp.up_proj",   d_model, d_ff),
            (f"block{block}.mlp.down_proj", d_ff,    d_model),
        ]:
            layer_index += 1
            total_linear_layers += 1
            p = params_linear(m_in, m_out, bias=True)
            m = macs_linear(m_in, m_out)
            total_params += p
            total_macs_per_token += m
            print(f"{layer_index:4d} | {proj_name:35s} | {m_in:6d} -> {m_out:6d} | {p:12d} | {m:12d}")

    # Embedding und Output-Projektion
    print("\nEmbedding and LM head (treated as linear):\n")
    for name, m_in, m_out in [
        ("tok_embedding", vocab_size, d_model),
        ("lm_head",       d_model,    vocab_size),
    ]:
        layer_index += 1
        total_linear_layers += 1
        p = params_linear(m_in, m_out, bias=False)
        m = macs_linear(m_in, m_out)
        total_params += p
        total_macs_per_token += m
        print(f"{layer_index:4d} | {name:35s} | {m_in:6d} -> {m_out:6d} | {p:12d} | {m:12d}")

    print("\nSummary (theoretical, from config):")
    print(f"  total linear-like layers   : {total_linear_layers}")
    print(f"  total parameters (approx.) : {total_params:,}")
    print(f"  total MACs per token       : {total_macs_per_token:,}")
    print(f"  approx FLOPs per token     : {2 * total_macs_per_token:,}")
    print("  (FLOPs ~= 2 * MACs: mul + add)")


def main():
    print(f"Loading config: {MODEL_NAME}")
    cfg = AutoConfig.from_pretrained(MODEL_NAME)  
    analyze_from_config(cfg)


if __name__ == "__main__":
    main()
