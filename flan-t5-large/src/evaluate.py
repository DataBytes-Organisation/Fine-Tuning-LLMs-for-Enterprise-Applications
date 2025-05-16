import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)

    avg_loss = total_loss / len(data_loader)
    cm = confusion_matrix(all_labels_tensor.numpy(), all_preds_tensor.numpy())

    results = {
        "loss": avg_loss,
        "confusion_matrix": cm,
        "all_preds": [all_preds_tensor],
        "all_labels": [all_labels_tensor]
    }

    return results
