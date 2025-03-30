from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import transformers

# Suppress logging
transformers.logging.set_verbosity_error()

# Load the model and tokenizer
model_name = "google/flan-t5-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define input text
input_text = "I drank alcohol during covid. What medicine should I take and explain why?"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt")

# Generate a response 
outputs = model.generate(inputs["input_ids"], max_length=50)

# Decode the output tokens to get the response text
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)