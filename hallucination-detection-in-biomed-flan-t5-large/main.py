import transformers
from model import load_model_and_tokenizer
from generate_response import generate_response

# Suppress logging
transformers.logging.set_verbosity_error()

# Load model and tokenizer
model_name = "google/flan-t5-large"
model, tokenizer = load_model_and_tokenizer(model_name)

# Define input text
input_text = "Is aspirin recommended for reducing high blood pressure during pregnancy?"

# Generate response
response = generate_response(model, tokenizer, input_text)

# Print response
print(response)
