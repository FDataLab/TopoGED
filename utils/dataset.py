import torch
from torch.utils.data import Dataset, DataLoader


# Update path for imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



# Dataset class for embeddings with flexibility for both train and test sets
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def __len__(self):
        return len(self.embeddings) - 1  # We want to predict the next embedding
    
    def __getitem__(self, idx):
        x = torch.Tensor(self.embeddings[idx]).unsqueeze(0)  # Add sequence dimension
        y = torch.Tensor(self.embeddings[idx + 1])  # Next embedding as target
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
