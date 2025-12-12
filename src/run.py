from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto"
    )

    text = "FFT-based acceleration for neural networks is"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model(**inputs)
    print("Forward pass successful. Logits shape:", outputs.logits.shape)

if __name__ == "__main__":
    main()
