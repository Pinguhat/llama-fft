import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from patch_llama_fft import MODEL_PATH, patch_mlp_with_block_circulant

TEXT = "FFT basierte Beschleunigung fuer neuronale Netze ist"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    inputs = tokenizer(TEXT, return_tensors="pt")

    # 1) Originalmodell
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
        logits_orig = model_orig(**inputs).logits

    # RAM aufraeumen
    del model_orig
    gc.collect()

    # 2) Gepatchtes Modell (mit BlockCirculantLinear)
    print("Loading FFT-patched model...")
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
        logits_fft = model_fft(**inputs).logits

    # 3) Unterschied messen (L2-Abstand der Logits)
    diff = (logits_orig - logits_fft).pow(2).mean().sqrt()
    print("L2-Abstand der Logits (Original vs. FFT-MLP):", diff.item())

if __name__ == "__main__":
    main()
