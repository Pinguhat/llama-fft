from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "/home/lukas/models/Llama-2-7b-hf"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        torch_dtype="float16",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()

    text = "FFT based acceleration for neural networks is"
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    print("Logits shape:", outputs.logits.shape)

if __name__ == "__main__":
    main()
