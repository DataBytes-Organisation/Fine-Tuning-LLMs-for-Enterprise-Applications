DEFAULT_MODEL_PATH = "./saved_model"

def save_model_and_tokenizer(model, tokenizer):
    model.save_pretrained(DEFAULT_MODEL_PATH)
    tokenizer.save_pretrained(DEFAULT_MODEL_PATH)
    print(f"Model and tokenizer saved to {DEFAULT_MODEL_PATH}")
