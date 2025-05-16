
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from config import ModelConfig
from transformers import AutoTokenizer
from src.data_loader import create_dataloaders
from src.model import create_model, count_trainable_parameters
from src.evaluate import evaluate_model

def train_model(model, train_loader, test_loader, config, device,tokenizer):
    all_labels = torch.cat([batch['labels'] for batch in train_loader])
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels.cpu().numpy()), y=all_labels.cpu().numpy())
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    best_val_loss = float('inf')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(config.output_dir, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{config.epochs}, Training Loss: {avg_train_loss:.4f}")

        results = evaluate_model(model, test_loader, device)
        val_loss = results['loss']
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
          
          # Save full model (overwrites previous best)
            best_val_loss = val_loss
            model.save_pretrained(session_dir)
            tokenizer.save_pretrained(session_dir)
            torch.save(model.state_dict(), os.path.join(session_dir, "best_model.pt"))
            # best_val_loss = val_loss
            # model_path = os.path.join(session_dir, f"best_model_epoch{epoch+1}.pt")
            # model.save_pretrained(session_dir)
            # torch.save(model.state_dict(), model_path)
            # tokenizer.save_pretrained(session_dir)
            print(f"Best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")

    # Plot training vs validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(session_dir, 'loss_plot.png'))
    plt.close()

    return model
