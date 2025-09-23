import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class NuclearDataset(Dataset):
    """Dataset class for nuclear segmentation and classification"""
    
    def __init__(self, 
                 image_paths: List[Path],
                 mask_paths: Optional[List[Path]] = None,
                 patch_size: Tuple[int, int] = (256, 256),
                 augment: bool = True,
                 normalize: bool = True):
        """
        Args:
            image_paths (List[Path]): List of paths to images
            mask_paths (Optional[List[Path]]): List of paths to masks
            patch_size (Tuple[int, int]): Size of image patches
            augment (bool): Whether to apply augmentations
            normalize (bool): Whether to normalize images
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.augment = augment
        self.normalize = normalize
        
        # Set up augmentation pipeline
        self.transform = self._get_transforms()
        
    def _get_transforms(self) -> A.Compose:
        """Get augmentation pipeline"""
        transforms = []
        
        if self.augment:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=45,
                    p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.5),
                ], p=0.3),
            ])
            
        if self.normalize:
            transforms.append(A.Normalize())
            
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask if available
        mask = None
        if self.mask_paths is not None:
            mask_path = self.mask_paths[idx]
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.0 if mask is not None else np.zeros_like(image[:,:,0])
        
        # Apply transforms
        if mask is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            return image, mask.unsqueeze(0)
        else:
            transformed = self.transform(image=image)
            image = transformed['image']
            return image, None

def create_patch_dataset(image_dir: Path,
                        mask_dir: Optional[Path] = None,
                        patch_size: Tuple[int, int] = (256, 256),
                        stride: Optional[Tuple[int, int]] = None) -> NuclearDataset:
   
    image_paths = sorted(Path(image_dir).glob('*.png'))
    mask_paths = None
    if mask_dir:
        mask_dir = Path(mask_dir)
        mask_paths = [mask_dir / f'{img_path.stem}_mask.png' for img_path in image_paths]
    
    # Use 75% overlap if stride not specified
    if stride is None:
        stride = (patch_size[0] // 4, patch_size[1] // 4)
    
    return NuclearDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        patch_size=patch_size
    )
