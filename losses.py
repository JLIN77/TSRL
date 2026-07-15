import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineConsistencyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        pred, target: [B,C,H,W]
        """

        B, C, H, W = pred.shape

        pred = pred.flatten(1)
        target = target.flatten(1)

        pred = F.normalize(pred, dim=1)
        target = F.normalize(target, dim=1)

        loss = 1 - (pred * target).sum(dim=1).mean()

        return loss

# ---------------------------------------------
# 4. TFPL Loss Wrapper
# ---------------------------------------------
class TFPLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = CosineConsistencyLoss()

    def forward(self, pred, target):
        return self.cos(pred, target)


# ---------------------------------------------
# 5. SFOL Loss Wrapper
# ---------------------------------------------
class SFOLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = CosineConsistencyLoss()

    def forward(self, obj_feat, high_feat):
        return self.cos(obj_feat, high_feat)