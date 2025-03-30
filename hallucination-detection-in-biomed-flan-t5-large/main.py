import transformers
from model import load_model_and_tokenizer
from generate_response import generate_response

# Suppress logging
transformers.logging.set_verbosity_error()

# Load model and tokenizer
model_name = "google/flan-t5-large"
model, tokenizer = load_model_and_tokenizer(model_name)

def main():
    print("Welcome to the Medical QA system. Ask your medical question (or type 'exit' to quit).")
    
    while True:
        # Ask the user for a question
        input_text = input("Please enter your medical question: ")
        
        # Exit condition
        if input_text.lower() == "exit":
            print("Exiting the Medical QA system. Goodbye!")
            break
        
        # Generate and display the answer
        response = generate_response(model, tokenizer, input_text)
        print("\nAnswer:", response)
        print("\n---\n")

if __name__ == "__main__":
    main()
