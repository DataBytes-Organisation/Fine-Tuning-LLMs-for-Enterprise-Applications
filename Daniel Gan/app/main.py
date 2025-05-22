from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.model import generate_analysis

app = FastAPI()

class InferenceRequest(BaseModel):
    input_text: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95

@app.post("/analyze")
def analyze(request: InferenceRequest):
    output = generate_analysis(
        input_data=request.input_text,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )
    return {"analysis": output}
