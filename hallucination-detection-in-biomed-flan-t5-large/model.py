from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_model_and_tokenizer(model_name="google/flan-t5-large"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
