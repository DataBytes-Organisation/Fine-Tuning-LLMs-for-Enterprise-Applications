
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

def tokenize_and_cache_data(datapath, tokenizer, max_length=128, cache_dir='cached_data_new'):
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = os.path.basename(datapath).replace('.csv', '_tokenized.pkl')
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        print(f"Loading cached tokenized data from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    df = pd.read_csv(datapath)
    prompt_prefix = "Classify the sentiment as positive, negative, or neutral: "
    texts = [prompt_prefix + str(text) for text in df['commentsReview'].tolist()]
    # texts = df['commentsReview'].tolist()
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    labels = [label_map[label.lower()] for label in df['sentiment']]

    encodings = tokenizer(texts, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    labels = torch.tensor(labels, dtype=torch.long)

    cached_data = (input_ids, attention_masks, labels)
    with open(cache_path, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"Tokenized data cached to {cache_path}")
    return cached_data

def create_dataloaders(config, tokenizer):
    train_input_ids, train_attention_masks, train_labels = tokenize_and_cache_data(config.train_data_path, tokenizer)
    test_input_ids, test_attention_masks, test_labels = tokenize_and_cache_data(config.test_data_path, tokenizer)

    train_dataset = SentimentDataset(train_input_ids, train_attention_masks, train_labels)
    test_dataset = SentimentDataset(test_input_ids, test_attention_masks, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader
