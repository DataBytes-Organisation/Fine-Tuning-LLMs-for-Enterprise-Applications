import transformers
from model import get_default_model_tokenizer, load_model_from_file
from generate_response import generate_response
from data_processing import load_and_tokenize_data
from trainer import train_model, evaluate_model
from utils import save_model_and_tokenizer

# Suppress logging
transformers.logging.set_verbosity_error()

def start_default_qa_system():
    model, tokenizer = get_default_model_tokenizer()
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

def retrain_model():    
    model, tokenizer = get_default_model_tokenizer()
    # Load and preprocess data
    train_encodings, train_labels = load_and_tokenize_data("./data/ori_pqal.json", tokenizer)
    
    # Train the model
    train_model(model, train_encodings, train_labels)

    # Evaluate the model - train data used as we do randomization any way
    evaluate_model(model, train_encodings, train_labels)

    # Save the model
    save_model_and_tokenizer(model, tokenizer)

def load_retrained_qa_system():
    model, tokenizer = load_model_from_file()
    print("Welcome to the retrained Medical QA system. Ask your medical question (or type 'exit' to quit).")
    
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

def main():
    print("Choose an option:")
    print("1: Load default model")
    print("2: Load retrained model")
    print("3: Train model")
    
    choice = input("Enter the number corresponding to your choice: ")
    
    if choice == '1':
        start_default_qa_system()
    elif choice == '2':
        load_retrained_qa_system()
    elif choice == '3':
        retrain_model()
    else:
        print("Invalid choice! Please enter a valid number.")

if __name__ == "__main__":
    main()
