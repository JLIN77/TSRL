import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        identity = x

        out = F.relu(self.norm(self.conv1(x)))
        out = self.norm(self.conv2(out))

        return F.relu(out + identity)