from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

import os
from transformers import AutoTokenizer

# Always resolve to absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "mistral-combined-finetuned-weights")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)
model.eval()

# Initialize FastAPI
app = FastAPI(title="Financial Risk Analysis API")

# Define request schema
class AnalysisRequest(BaseModel):
    input_text: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95

# Define your inference function
def generate_analysis(input_data: str, max_new_tokens=512, temperature=0.7, top_p=0.95):
    prompt = (
        "You are an expert financial risk analyst. Analyze the provided text for financial risks, "
        "and output a structured assessment in JSON format including risk detection, specific risk flags, "
        "financial exposure details, and analysis notes. "
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

# Define the endpoint
@app.post("/analyze")
def analyze_risk(request: AnalysisRequest):
    output = generate_analysis(
        input_data=request.input_text,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )
    return {"analysis": output}
