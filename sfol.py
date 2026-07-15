import torch
import torch.nn as nn
import torch.nn.functional as F


class SFOL(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.attn = SelfAttention(dim)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1),
        )

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, bev_low, bev_high):
        """
        bev_low : low beam BEV
        bev_high: high beam BEV (teacher)
        """

        x = self.attn(bev_low)

        x = self.norm1(x + bev_low)

        x = self.mlp(x)

        bev_obj = self.norm2(x + bev_low)

        return bev_obj