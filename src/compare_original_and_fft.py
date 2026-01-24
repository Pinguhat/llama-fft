import gc
import torch
import torch.nn.functional as F
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
    # 1) Original model 
    print("Loading original model...")
    model_orig = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model_orig.eval()

    def run_model(model, text):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            return model(**inputs).logits

    logits_orig_list = [run_model(model_orig, t) for t in TEXTS]

    # clean RAM
    del model_orig
    gc.collect()

    # 2) FFT-patched model
    print("Loading FFT-patched model...")
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
    #load_bc_params(model_fft, "llama-fft/src/bc_calibrated.pt")

    logits_fft_list = [run_model(model_fft, t) for t in TEXTS]

    l2_vals, kl_vals, cos_vals = [], [], []

    top1_match = 0
    topk_overlap_vals = []

    for lo, lf in zip(logits_orig_list, logits_fft_list):
        l2_vals.append((lo - lf).pow(2).mean().sqrt().item())

        # KL(teacher || student): teacher=orig, student=fft
        logp_student = F.log_softmax(lf[:, -1, :].to(torch.float32), dim=-1)
        p_teacher = F.softmax(lo[:, -1, :].to(torch.float32), dim=-1)
        kl_vals.append(F.kl_div(logp_student, p_teacher, reduction="batchmean").item())

        a = lo[:, -1, :].to(torch.float32)
        b = lf[:, -1, :].to(torch.float32)
        cos_vals.append(F.cosine_similarity(a, b, dim=-1).mean().item())

        # Top-1 match on last token
        next_id_orig = int(lo[0, -1].argmax(dim=-1).item())
        next_id_fft = int(lf[0, -1].argmax(dim=-1).item())
        top1_match += int(next_id_orig == next_id_fft)

        # Top-k overlap on last token (robust agreement)
        topk_orig = torch.topk(lo[0, -1], k=TOPK, dim=-1).indices.tolist()
        topk_fft = torch.topk(lf[0, -1], k=TOPK, dim=-1).indices.tolist()
        overlap = len(set(topk_orig).intersection(set(topk_fft)))
        topk_overlap_vals.append(overlap)

    print("Texts tested:", len(TEXTS))
    print("L2 mean/std:", float(torch.tensor(l2_vals).mean()), float(torch.tensor(l2_vals).std()))
    print("KL mean/std:", float(torch.tensor(kl_vals).mean()), float(torch.tensor(kl_vals).std()))
    print("Cos mean/std:", float(torch.tensor(cos_vals).mean()), float(torch.tensor(cos_vals).std()))

    print(f"Top-1 next-token match: {top1_match} / {len(TEXTS)}")
    print(f"Top-{TOPK} next-token overlap mean/std:",
          float(torch.tensor(topk_overlap_vals, dtype=torch.float32).mean()),
          float(torch.tensor(topk_overlap_vals, dtype=torch.float32).std()))

if __name__ == "__main__":
    main()
