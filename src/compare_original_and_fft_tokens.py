import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from patch_llama_fft import MODEL_PATH, patch_mlp_with_block_circulant

TEXT = "FFT basierte Beschleunigung fuer neuronale Netze ist"

def main():
    # 1) Tokenizer und Input vorbereiten (nur einmal)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    inputs = tokenizer(TEXT, return_tensors="pt")

    # 2) Originalmodell: Logits + Top-1 Tokens
    print("Loading original model...")
    model_orig = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model_orig.eval()

    with torch.no_grad():
        outputs_orig = model_orig(**inputs)
        logits_orig = outputs_orig.logits  # shape: (1, seq_len, vocab_size)

    pred_tokens_orig = logits_orig.argmax(dim=-1)  # (1, seq_len)
    print("Original predicted token ids:", pred_tokens_orig.tolist())

    # 3) RAM aufraeumen
    del model_orig
    gc.collect()

    # 4) FFT-gepatchtes Modell: Logits + Top-1 Tokens
    print("\nLoading FFT-patched model...")
    model_fft = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model_fft.eval()
    patch_mlp_with_block_circulant(model_fft)

    with torch.no_grad():
        outputs_fft = model_fft(**inputs)
        logits_fft = outputs_fft.logits

    pred_tokens_fft = logits_fft.argmax(dim=-1)
    print("FFT-MLP predicted token ids:", pred_tokens_fft.tolist())

    # 5) Vergleich: Wie viele Positions-Vorhersagen sind gleich?
    same = (pred_tokens_orig == pred_tokens_fft).sum().item()
    total = pred_tokens_orig.numel()
    print(f"\nSame predicted tokens: {same} / {total}")

    # 6) Optional: Next-token Prediction fuer letztes Token vergleichen

    # Wir nehmen die Logits der letzten Position (seq_len-1)
    last_logits_orig = logits_orig[0, -1]  # (vocab_size,)
    last_logits_fft = logits_fft[0, -1]

    next_id_orig = last_logits_orig.argmax(dim=-1).item()
    next_id_fft = last_logits_fft.argmax(dim=-1).item()

    next_tok_orig = tokenizer.decode([next_id_orig])
    next_tok_fft = tokenizer.decode([next_id_fft])

    print("\nNext-token prediction (based on last position):")
    print("  Original top-1 id  :", next_id_orig, "| token:", repr(next_tok_orig))
    print("  FFT-MLP top-1 id   :", next_id_fft, "| token:", repr(next_tok_fft))

if __name__ == "__main__":
    main()
