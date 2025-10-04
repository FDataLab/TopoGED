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
    

# Dataset class that allows us to predict the change in embedding from yesterday, then add it to the previous days embedding
class DeltaEmbeddingDataset(Dataset):
    def __init__(self, embeddings, k):
        self.embeddings = embeddings
        self.k = k

    def __len__(self):
        return len(self.embeddings) - self.k

    def __getitem__(self, idx):
        # Input: A sequence of 'k' vectors.
        x = self.embeddings[idx : idx + self.k]
        
        # Target: The delta (change) between the next vector and the last vector in the input sequence.
        y = self.embeddings[idx + self.k] - self.embeddings[idx + self.k - 1]
        
        # We also grab the last vector of the input sequence.
        x_last = self.embeddings[idx + self.k - 1]
        
        # Convert to PyTorch tensors and return the three values.
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        x_last = torch.tensor(x_last, dtype=torch.float32)

        return x, y, x_last
