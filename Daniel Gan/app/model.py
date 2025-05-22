import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model (already merged weights)
model_path = "./mistral-combined-finetuned-weights"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def generate_analysis(input_data: str, max_new_tokens=512, temperature=0.7, top_p=0.95):
    prompt = (
        "You are an expert financial risk analyst. Analyze the provided text for financial risks, "
        "and output a structured assessment in JSON format including risk detection, specific risk "
        "flags, financial exposure details, and analysis notes. "
        f"{input_data}"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
        end = time.time()

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n⏱️ Generation time: {end - start:.2f} seconds")
    return result


