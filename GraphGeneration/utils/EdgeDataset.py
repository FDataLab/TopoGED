import torch
from torch.utils.data import Dataset, DataLoader

class EdgeDataset(Dataset):
    def __init__(self, u_array, v_array, y_array):
        # Handle cases where slice might be empty
        self.u_ids = torch.from_numpy(u_array).long() if len(u_array) > 0 else torch.empty(0).long()
        self.v_ids = torch.from_numpy(v_array).long() if len(v_array) > 0 else torch.empty(0).long()
        self.labels = torch.from_numpy(y_array).float() if len(y_array) > 0 else torch.empty(0).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.u_ids[idx], self.v_ids[idx], self.labels[idx]