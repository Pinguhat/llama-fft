import gc
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from patch_llama_fft import MODEL_PATH, patch_mlp_with_block_circulant, load_bc_params


TEXTS = [
    "FFT basierte Beschleunigung fuer neuronale Netze ist",
    "In diesem Paper untersuchen wir Block-Circulant Matrices fuer LLMs.",
    "Die Genauigkeit wird ueber Logits-Distanzen und Token-Agreement bewertet.",
]

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
    load_bc_params(model_fft, "llama-fft/src/bc_calibrated.pt")

    logits_fft_list = [run_model(model_fft, t) for t in TEXTS]

    l2_vals, kl_vals, cos_vals = [], [], []

    for lo, lf in zip(logits_orig_list, logits_fft_list):
        l2_vals.append((lo - lf).pow(2).mean().sqrt().item())

        # KL(teacher || student): teacher=orig, student=fft
        logp_student = F.log_softmax(lf[:, -1, :].to(torch.float32), dim=-1)
        p_teacher = F.softmax(lo[:, -1, :].to(torch.float32), dim=-1)
        kl_vals.append(F.kl_div(logp_student, p_teacher, reduction="batchmean").item())

        a = lo[:, -1, :].to(torch.float32)
        b = lf[:, -1, :].to(torch.float32)
        cos_vals.append(F.cosine_similarity(a, b, dim=-1).mean().item())

    print("Texts tested:", len(TEXTS))
    print("L2 mean/std:", float(torch.tensor(l2_vals).mean()), float(torch.tensor(l2_vals).std()))
    print("KL mean/std:", float(torch.tensor(kl_vals).mean()), float(torch.tensor(kl_vals).std()))
    print("Cos mean/std:", float(torch.tensor(cos_vals).mean()), float(torch.tensor(cos_vals).std()))

if __name__ == "__main__":
    main()
