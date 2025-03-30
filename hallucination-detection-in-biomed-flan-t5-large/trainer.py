import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from data_processing import load_and_tokenize_data
import random
from transformers import Adafactor
from peft import get_peft_model, LoraConfig, TaskType

def train_model(model, encodings, labels, num_samples=20):
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
    for epoch in range(5):  # Adjust number of epochs as needed
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
        
        print(f"Epoch {epoch + 1} Loss: {loss.item()}")

def train_model_lora(model, encodings, labels, num_samples=5):
    # Apply LoRA
    lora_config = LoraConfig(
        r=8,                    # Rank (adjust for trade-off between speed & accuracy)
        lora_alpha=16,           # Scaling factor
        lora_dropout=0.05,       # Regularization
        task_type=TaskType.SEQ_2_SEQ_LM  # Task type (adjust if necessary)
    )
    model = get_peft_model(model, lora_config)

    # Randomly sample `num_samples` data
    sampled_indices = random.sample(range(len(encodings['input_ids'])), num_samples)
    encodings = {key: torch.stack([value[i] for i in sampled_indices]) for key, value in encodings.items()}
    labels = torch.stack([labels[i] for i in sampled_indices])

    # Convert to tensors for DataLoader
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Prepare DataLoader for batching
    train_dataset = TensorDataset(input_ids, attention_mask, labels)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

    model.train()
    for epoch in range(5):  # Adjust number of epochs as needed
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
