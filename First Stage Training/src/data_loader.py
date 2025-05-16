
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config import ModelConfig
import pickle

class SentimentDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

def tokenize_and_cache_data(datapath, tokenizer, max_length=128, cache_dir='/content/drive/My Drive/Deakin_units/T1-2025/SIT764/sentiment-analysis/cached_data'):
    """
    Tokenize data and save to cache for faster loading
    
    Args:
    - datapath (str): Path to CSV file
    - tokenizer (AutoTokenizer): Tokenizer to use
    - max_length (int): Maximum sequence length
    - cache_dir (str): Directory to save cached files
    
    Returns:
    - Tuple of (input_ids, attention_masks, labels)
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filename
    cache_filename = os.path.basename(datapath).replace('.csv', '_tokenized.pkl')
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Check if cached file exists
    if os.path.exists(cache_path):
        print(f"Loading cached tokenized data from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Read CSV
    df = pd.read_csv(datapath)
    
    # Prepare texts and labels
    texts = df['commentsReview'].tolist()
    
    # Map labels to integers
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    labels = [label_map[label.lower()] for label in df['sentiment']]
    
    # Tokenize texts
    encodings = tokenizer(
        texts, 
        return_tensors='pt', 
        max_length=max_length, 
        padding='max_length', 
        truncation=True
    )
    
    # Extract input_ids and attention_masks
    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    
    # Convert to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Cache the tokenized data
    cached_data = (input_ids, attention_masks, labels)
    with open(cache_path, 'wb') as f:
        pickle.dump(cached_data, f)
    
    print(f"Tokenized data cached to {cache_path}")
    return cached_data

def create_dataloaders(config, tokenizer):
    # Tokenize and cache train data
    train_input_ids, train_attention_masks, train_labels = tokenize_and_cache_data(
        config.train_data_path, 
        tokenizer
    )
    
    # Tokenize and cache test data
    test_input_ids, test_attention_masks, test_labels = tokenize_and_cache_data(
        config.test_data_path, 
        tokenizer
    )
    
    # Create datasets
    train_dataset = SentimentDataset(
        train_input_ids, 
        train_attention_masks, 
        train_labels
    )
    test_dataset = SentimentDataset(
        test_input_ids, 
        test_attention_masks, 
        test_labels
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader
