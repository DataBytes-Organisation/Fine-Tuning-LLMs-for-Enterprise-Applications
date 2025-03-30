from transformers import AutoTokenizer
import json

def load_and_tokenize_data(file_path, tokenizer):
    data = load_data(file_path)

    questions = []
    contexts = []
    long_answers = []

    for item_id, item in data.items():
        # Ensure that the item contains a valid 'QUESTION' field        
        if 'QUESTION' in item and item['QUESTION']:
            # Only add to lists if 'QUESTION' and 'CONTEXTS' are present
            if 'CONTEXTS' in item and item['CONTEXTS']:
                questions.append(item['QUESTION'])
                contexts.append(" ".join(item['CONTEXTS']))  # Combine context into a single string
                long_answers.append(item.get('LONG_ANSWER', ''))  # Get LONG_ANSWER, default to empty if not present
            else:
                print(f"Skipping question due to missing context: {item['QUESTION']}")
        else:
            print(f"Skipping item due to missing or empty QUESTION: {item}") 

    # Tokenize question and context together
    # encodings = tokenizer(questions, padding=True, truncation=True, return_tensors="pt", max_length=512)
    encodings = tokenizer(questions, contexts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    # Tokenize answers for labels (ensure answers are not truncated unnecessarily)
    labels = tokenizer(long_answers, padding=True, truncation=True, return_tensors="pt", max_length=512)["input_ids"]
    
    # Make sure to set the labels to -100 where we want the model to ignore padding during loss calculation
    labels[labels == tokenizer.pad_token_id] = -100

    return encodings, labels

# Function to load data
def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)