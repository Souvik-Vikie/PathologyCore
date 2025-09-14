import torch
import numpy as np
from typing import Dict, Optional

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Dice loss
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor
        
    Returns:
        torch.Tensor: Dice loss value
    """
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1.0 - dice

def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Dice score
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor
        
    Returns:
        torch.Tensor: Dice score value
    """
    pred = (pred > 0.5).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate IoU (Intersection over Union) score
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor
        
    Returns:
        torch.Tensor: IoU score value
    """
    pred = (pred > 0.5).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

class DiceBCELoss(torch.nn.Module):
    """Combined Dice and BCE loss"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights (Optional[Dict[str, float]]): Weights for BCE and Dice losses
                                                Default: {'bce': 0.5, 'dice': 0.5}
        """
        super().__init__()
        self.weights = weights or {'bce': 0.5, 'dice': 0.5}
        self.bce = torch.nn.BCEWithLogitsLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = dice_loss(pred, target)
        bce = self.bce(pred, target)
        return self.weights['bce'] * bce + self.weights['dice'] * dice
