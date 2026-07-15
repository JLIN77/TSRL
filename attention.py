import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        B, C, H, W = x.shape
        N = H * W

        q = self.q(x).flatten(2).transpose(1, 2)
        k = self.k(x).flatten(2)
        v = self.v(x).flatten(2).transpose(1, 2)

        attn = torch.bmm(q, k) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)

        return out.transpose(1, 2).reshape(B, C, H, W)