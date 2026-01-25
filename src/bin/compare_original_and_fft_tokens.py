import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from patch_llama_fft import MODEL_PATH, patch_mlp_with_block_circulant, load_bc_params


TEXTS = [
"Explain in one sentence why approximation errors can accumulate in deep networks.",
"Complete the sentence: A block-circulant projection changes the model because",
"Continue: When logits shift slightly, the most likely token can",
"Finish this thought: Using FFTs helps mainly by reducing",
"In plain words, describe what KL divergence measures for next-token distributions.",
"Continue: A practical trade-off in model compression is",
"Write the next phrase: The main advantage of structured matrices is",
"Complete: If top-1 changes but top-5 stays similar, then",
"Continue: One reason cosine similarity is useful is that it",
"Finish the sentence: Calibration after approximation is similar to",
"Continue: In transformers, the MLP layers are important because they",
"Complete: The risk of aggressive compression is that it may",
"Continue: A simple sanity check for a patched model is",
"Finish this thought: The difference between L2 and KL on logits is",
"Continue: If the teacher distribution is sharp, small perturbations can",
"Vervollständige: Kleine Logit-Verschiebungen können dazu führen, dass",
"Führe fort: Ein Vorteil strukturierter Gewichtsmatrizen ist, dass",
"Erkläre kurz: Wieso kann sich ein Fehler in frühen Layern später verstärken?",
"Vervollständige: Eine Kalibrierung nach der Approximation soll erreichen, dass",
"Führe fort: Ein einfacher Test für die Ähnlichkeit zweier Modelle ist",
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
    # load_bc_params(model_fft, "llama-fft/src/bc_calibrated.pt")
    # load_bc_params(model_fft, r"..\llama-fft\src\bc_calibrated.pt")
    total_same = 0
    total_tokens = 0

    total_next_same = 0
    total_next_topk = 0

    per_text_agree_ratios = []
    next_topk_overlap_vals = []

    for (t, logits_orig, pred_orig, input_ids) in orig_runs:
        logits_fft, _ = run(model_fft, t)
        pred_fft = logits_fft.argmax(dim=-1)

        same = (pred_orig == pred_fft).sum().item()
        total = pred_orig.numel()
        total_same += same
        total_tokens += total

        per_text_agree_ratios.append(same / max(total, 1))

        # Top-1 next token
        next_id_orig = int(logits_orig[0, -1].argmax(dim=-1).item())
        next_id_fft = int(logits_fft[0, -1].argmax(dim=-1).item())
        total_next_same += int(next_id_orig == next_id_fft)

        # Top-k next token: is original top-1 within patched top-k?
        topk_ids_fft = torch.topk(logits_fft[0, -1], k=TOPK, dim=-1).indices.tolist()
        in_topk = int(next_id_orig in topk_ids_fft)
        total_next_topk += in_topk

        # Top-k overlap (orig vs patched) on next token distribution
        topk_ids_orig = torch.topk(logits_orig[0, -1], k=TOPK, dim=-1).indices.tolist()
        overlap = len(set(topk_ids_orig).intersection(set(topk_ids_fft)))
        next_topk_overlap_vals.append(overlap)

        print("\nText:", repr(t))
        print(f"  Same tokens: {same} / {total}")
        print("  Next-token ids:", next_id_orig, "vs", next_id_fft)
        print("  Next-token:", repr(tokenizer.decode([next_id_orig])), "vs", repr(tokenizer.decode([next_id_fft])))
        print(f"  Next-token in top-{TOPK} (patched):", bool(in_topk))
        print(f"  Next-token top-{TOPK} overlap (orig vs patched):", overlap, "/", TOPK)
        if not in_topk:
            print("  Patched top-k:", [repr(tokenizer.decode([i])) for i in topk_ids_fft])

    print("\nOverall token agreement:", total_same, "/", total_tokens)
    print("Overall next-token agreement:", total_next_same, "/", len(TEXTS))
    print(f"Overall next-token top-{TOPK} agreement:", total_next_topk, "/", len(TEXTS))

    print("Per-text token agreement mean/std:",
          float(torch.tensor(per_text_agree_ratios, dtype=torch.float32).mean()),
          float(torch.tensor(per_text_agree_ratios, dtype=torch.float32).std()))
    print(f"Next-token top-{TOPK} overlap mean/std:",
          float(torch.tensor(next_topk_overlap_vals, dtype=torch.float32).mean()),
          float(torch.tensor(next_topk_overlap_vals, dtype=torch.float32).std()))

if __name__ == "__main__":
    main()
