import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def plot_training_curves(metrics: Dict[str, List[float]], save_path: Optional[Path] = None):
    """
    Plot training metrics over epochs
    
    Args:
        metrics (Dict[str, List[float]]): Dictionary of metrics
        save_path (Optional[Path]): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Metrics')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def overlay_masks(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create an overlay of image and segmentation mask
    
    Args:
        image (np.ndarray): Original image
        mask (np.ndarray): Binary segmentation mask
        alpha (float): Transparency of the overlay
        
    Returns:
        np.ndarray: Overlay image
    """
    mask_rgb = np.zeros_like(image)
    mask_rgb[:, :, 1] = mask * 255  # Green channel
    overlay = cv2.addWeighted(image, 1-alpha, mask_rgb, alpha, 0)
    return overlay

def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Calculate various performance metrics
    
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    pred = pred > 0.5
    pred = pred.float()
    
    # Calculate intersection and union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    # Calculate metrics
    dice = (2.0 * intersection) / (pred.sum() + target.sum() + 1e-6)
    iou = intersection / (union + 1e-6)
    precision = intersection / (pred.sum() + 1e-6)
    recall = intersection / (target.sum() + 1e-6)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metrics: Dict[str, float],
                   save_path: Path):
    """
    Save model checkpoint
    
    Args:
        model (torch.nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer state
        epoch (int): Current epoch
        metrics (Dict[str, float]): Current metrics
        save_path (Path): Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   checkpoint_path: Path) -> Tuple[int, Dict[str, float]]:
    """
    Load model checkpoint
    
    Args:
        model (torch.nn.Module): Model to load weights into
        optimizer (torch.optim.Optimizer): Optimizer to load state into
        checkpoint_path (Path): Path to checkpoint file
        
    Returns:
        Tuple[int, Dict[str, float]]: Epoch number and metrics
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']
