import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphletLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions_graphlet, target_graphlet):
        cos_sim = F.cosine_similarity(predictions_graphlet, target_graphlet, dim=-1)  
        loss = 1 - cos_sim 
        return loss.mean() 