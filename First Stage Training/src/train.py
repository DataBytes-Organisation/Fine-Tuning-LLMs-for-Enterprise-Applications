
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(train_loader)

def train_model(model, train_loader, test_loader, config, device):
    # Prepare optimizer and schedule
    optimizer = AdamW(
        model.parameters(), 
        lr=config.learning_rate
    )
    
    # Total training steps
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    best_accuracy = 0
    
    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}/{config.epochs}")
        
        # Training
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        
        # Evaluation
        val_results = evaluate_model(
            model, test_loader, device
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
        
        # Save best model
        if val_results['accuracy'] > best_accuracy:
            best_accuracy = val_results['accuracy']
            model.save_pretrained(config.output_dir)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
    
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Import evaluation metrics here to avoid circular imports
    from sklearn.metrics import (
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score, 
        confusion_matrix
    )
    
    results = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1_score': f1_score(all_labels, all_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    
    return results
