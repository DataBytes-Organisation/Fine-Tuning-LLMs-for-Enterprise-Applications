import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from data_processing import load_and_tokenize_data
import random

def train_model(model, encodings, labels, num_samples=5):
     # Randomly sample `num_samples` data
    sampled_indices = random.sample(range(len(encodings['input_ids'])), num_samples)
    encodings = {key: value[sampled_indices] for key, value in encodings.items()}
    labels = labels[sampled_indices]

    # Convert to tensors for DataLoader
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    # Prepare DataLoader for batching
    train_dataset = TensorDataset(input_ids, attention_mask, labels)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(3):  # Adjust number of epochs as needed
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
        
        print(f"Epoch {epoch + 1} Loss: {loss.item()}")


def evaluate_model(model, encodings, labels, num_samples=5):
    # Randomly sample `num_samples` data
    sampled_indices = random.sample(range(len(encodings['input_ids'])), num_samples)
    encodings = {key: value[sampled_indices] for key, value in encodings.items()}
    labels = labels[sampled_indices]

    # Convert to tensors for DataLoader
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Prepare DataLoader for batching
    eval_dataset = TensorDataset(input_ids, attention_mask, labels)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

    model.eval()
    total_loss = 0
    correct = 0  # For counting the number of correct predictions
    total = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Get predicted answers (assuming model returns logits for Q&A)
            predictions = outputs.logits.argmax(dim=-1)

            # Calculate accuracy (simple token-by-token comparison)
            for i in range(len(labels)):
                total += 1
                if torch.equal(predictions[i], labels[i]):
                    correct += 1

    # Print the evaluation results
    avg_loss = total_loss / len(eval_loader)
    accuracy = correct / total

    print(f"Evaluation Loss: {avg_loss}")
    print(f"Accuracy: {accuracy}")
