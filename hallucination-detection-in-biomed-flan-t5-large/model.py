from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import DEFAULT_MODEL_PATH

def load_model_and_tokenizer(model_name="google/flan-t5-large"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_default_model_tokenizer():
    model_name = "google/flan-t5-large"
    model, tokenizer = load_model_and_tokenizer(model_name)
    return model, tokenizer

def load_model_from_file(model_path=DEFAULT_MODEL_PATH):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer