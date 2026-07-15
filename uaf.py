import torch
import torch.nn as nn
import torch.nn.functional as F

class UAF(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, bev_obj, bev_pred):

        """
        bev_obj : [B,C,H,W]
        bev_pred: [B,C,H,W]
        """

        # variance estimation (channel-wise)
        var_obj = bev_obj.var(dim=1, keepdim=True)
        var_pred = bev_pred.var(dim=1, keepdim=True)

        K = var_obj / (var_obj + var_pred + 1e-6)

        bev_t = bev_pred + K * (bev_obj - bev_pred)

        return bev_t, K