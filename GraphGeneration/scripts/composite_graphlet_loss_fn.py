import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphletLoss(nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_hi=1.0, lambda_cdf=1.0, eps=1e-8):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_hi = lambda_hi
        self.lambda_cdf = lambda_cdf
        self.eps = eps

    def forward(self, pred_counts, true_counts):
        """
        Inputs:
            pred_counts: Tensor of shape [batch_size, d] - raw graphlet counts from predicted graph
            true_counts: Tensor of shape [batch_size, d] - raw graphlet counts from ground-truth graph
        """
        pred_counts = pred_counts.float()
        true_counts = true_counts.float()

        # Normalize to graphlet distributions
        pred_sum = pred_counts.sum(dim=1, keepdim=True) + self.eps
        true_sum = true_counts.sum(dim=1, keepdim=True) + self.eps

        pred_dist = pred_counts / pred_sum
        true_dist = true_counts / true_sum

        # 1. MSE Loss
        mse_loss = F.mse_loss(pred_dist, true_dist, reduction='mean')

        # 2. Histogram Intersection Loss
        intersection = torch.minimum(pred_dist, true_dist).sum(dim=1)  # [batch_size]
        hi_loss = 1.0 - intersection.mean()

        # 3. Cumulative L1 Distance (Wasserstein-1)
        pred_cdf = torch.cumsum(pred_dist, dim=1)
        true_cdf = torch.cumsum(true_dist, dim=1)
        cdf_loss = F.l1_loss(pred_cdf, true_cdf, reduction='mean')

        # Final weighted loss
        total_loss = (
            self.lambda_mse * mse_loss +
            self.lambda_hi * hi_loss +
            self.lambda_cdf * cdf_loss
        )
        return total_loss
