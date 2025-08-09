import torch
from torch.utils.data import Dataset, DataLoader


# Update path for imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



# Dataset class for embeddings with flexibility for both train and test sets
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, k):
        self.embeddings = embeddings  # shape: [num_days, features]
        self.k = k

    def __len__(self):
        return len(self.embeddings) - self.k

    def __getitem__(self, idx):
        x = self.embeddings[idx : idx + self.k]       # shape: [k, features]
        y = self.embeddings[idx + self.k]             # shape: [features]
        return x, y
    
class BinaryDataset(Dataset):
    def __init__(self, embeddings, truth):
        self.embeddings = embeddings
        self.truth = truth
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.embeddings[idx]) 
        y = torch.LongTensor([self.truth[idx]])
        return x, y
