import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from patch_llama_fft import MODEL_PATH, patch_mlp_with_block_circulant, load_bc_params


TEXTS = [
    "Complete the sentence: FFT-based acceleration is useful because",
    "Complete the sentence: Block-circulant weights reduce cost by",
    "Finish this thought: The main downside of approximation is that",
    "Finish this thought: A practical way to recover accuracy is to",
    "Continue: KL divergence is a good metric here since",
    "Continue: Cosine similarity is helpful because it",
    "Continue: Small logit shifts can flip argmax when",
    "Continue: A common symptom of drift is that",
    "Continue: The goal of a least-squares projection is to",
    "Continue: In a transformer MLP, the gate projection typically",
    "Continue: The up projection increases dimension so that",
    "Continue: The down projection reduces dimension to",
    "Continue: With block size 256, we expect that",
    "Continue: With block size 64, we expect that",
    "Fill in one word only (no punctuation): between",
    "Vervollstaendige den Satz: FFT-Beschleunigung ist nuetzlich, weil",
    "Vervollstaendige den Satz: Block-Circulant Gewichte sparen Rechenzeit, indem",
    "Fuehre den Gedanken fort: Ein Nachteil der Approximation ist, dass",
    "Fuehre den Gedanken fort: Eine sinnvolle Metrik ist KL, weil",
    "Gib genau ein Wort als Fortsetzung (ohne Satzzeichen): zwischen",
]

TOPK = 5

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    print("Loading original model...")
    model_orig = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model_orig.eval()

    def run(model, text):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        return logits, inputs["input_ids"]

    orig_runs = []
    for t in TEXTS:
        logits, input_ids = run(model_orig, t)
        pred = logits.argmax(dim=-1)
        orig_runs.append((t, logits, pred, input_ids))

    del model_orig
    gc.collect()

    print("\nLoading FFT-patched model...")
    model_fft = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model_fft.eval()
    patch_mlp_with_block_circulant(model_fft)
    # Load calibrated BC params
    load_bc_params(model_fft, "llama-fft/src/bc_calibrated.pt")

    total_same = 0
    total_tokens = 0

    total_next_same = 0
    total_next_topk = 0

    for (t, logits_orig, pred_orig, input_ids) in orig_runs:
        logits_fft, _ = run(model_fft, t)
        pred_fft = logits_fft.argmax(dim=-1)

        same = (pred_orig == pred_fft).sum().item()
        total = pred_orig.numel()
        total_same += same
        total_tokens += total

        # Top-1 next token
        next_id_orig = int(logits_orig[0, -1].argmax(dim=-1).item())
        next_id_fft = int(logits_fft[0, -1].argmax(dim=-1).item())
        total_next_same += int(next_id_orig == next_id_fft)

        # Top-k next token: is original top-1 within patched top-k?
        topk_ids_fft = torch.topk(logits_fft[0, -1], k=TOPK, dim=-1).indices.tolist()
        in_topk = int(next_id_orig in topk_ids_fft)
        total_next_topk += in_topk

        print("\nText:", repr(t))
        print(f"  Same tokens: {same} / {total}")
        print("  Next-token ids:", next_id_orig, "vs", next_id_fft)
        print("  Next-token:", repr(tokenizer.decode([next_id_orig])), "vs", repr(tokenizer.decode([next_id_fft])))
        print(f"  Next-token in top-{TOPK} (patched):", bool(in_topk))
        if not in_topk:
            print("  Patched top-k:", [repr(tokenizer.decode([i])) for i in topk_ids_fft])

    print("\nOverall token agreement:", total_same, "/", total_tokens)
    print("Overall next-token agreement:", total_next_same, "/", len(TEXTS))
    print(f"Overall next-token top-{TOPK} agreement:", total_next_topk, "/", len(TEXTS))

if __name__ == "__main__":
    main()
