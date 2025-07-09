def generate_response(model, tokenizer, input_text, max_length=50):    
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
      inputs["input_ids"],
      max_length=max_length
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_parameterized_response(model, tokenizer, input_text, max_length=100, temperature=0.3):

    prompt = f"""
      Answer the following medical question using evidence-based clinical guidelines from trusted sources like PubMed, FDA, ACOG, WHO, etc. Provide relevant recommendations from these sources and explain the reasoning behind the answer. Include any conditions, risks, or precautions, and clearly state if further consultation is needed. If no evidence is available, explain the limitations of the available data.

      Question: {input_text}
    """
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
      inputs["input_ids"],
      max_length=max_length,
      temperature=temperature,
      do_sample=True, # Disables randomness for more reliable outputs
      num_beams=5, # Increases search space for high-quality answers
      repetition_penalty=1.2,  # Reduces hallucination by discouraging word overuse
      length_penalty=1.0,  # Maintains natural sentence flow
      early_stopping=True,  # Stops generation when an answer is complete
      num_return_sequences=1  # Returns only the best response
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
