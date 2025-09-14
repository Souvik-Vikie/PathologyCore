from pathlib import Path
from typing import Dict, Optional, Union, List
import torch
import numpy as np
import cv2
from .segmentation.unet import UNet
from .classification.classifier import NuclearClassifier
from .quantification.analyzer import NuclearAnalyzer
from .utils.visualization import overlay_masks
import yaml
import logging

logger = logging.getLogger(__name__)

class NuclearAnalysisPipeline:
    """End-to-end pipeline for nuclear analysis"""
    
    def __init__(self, 
                 seg_model_path: Union[str, Path],
                 cls_model_path: Optional[Union[str, Path]] = None,
                 config_path: Optional[Union[str, Path]] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the pipeline
        
        Args:
            seg_model_path: Path to segmentation model weights
            cls_model_path: Optional path to classification model weights
            config_path: Optional path to pipeline configuration
            device: Device to run models on
        """
        self.device = device
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize models
        self.seg_model = self._init_segmentation_model(seg_model_path)
        self.cls_model = self._init_classification_model(cls_model_path) if cls_model_path else None
        
        # Initialize analyzers
        self.nuclear_analyzer = NuclearAnalyzer()
        
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_segmentation_model(self, model_path: Union[str, Path]) -> UNet:
        """Initialize and load segmentation model"""
        model = UNet(n_channels=3, n_classes=1)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def _init_classification_model(self, model_path: Union[str, Path]) -> NuclearClassifier:
        """Initialize and load classification model"""
        model = NuclearClassifier()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Normalize and convert to tensor
        image = image / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image = image.unsqueeze(0)
        return image.to(self.device)
    
    def process_image(self, 
                     image_path: Union[str, Path],
                     save_dir: Optional[Union[str, Path]] = None) -> Dict:
        """
        Process a single image through the pipeline
        
        Args:
            image_path: Path to input image
            save_dir: Optional directory to save results
            
        Returns:
            Dict containing analysis results
        """
        # Setup saving
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        model_input = self.preprocess_image(image)
        
        # Generate segmentation
        with torch.no_grad():
            pred = self.seg_model(model_input)
            pred_mask = (torch.sigmoid(pred) > 0.5).cpu().numpy()[0, 0]
        
        # Analyze nuclei
        nuclei_features = self.nuclear_analyzer.analyze_morphology(pred_mask)
        density = self.nuclear_analyzer.calculate_density(nuclei_features, pred_mask.shape)
        
        # Classify nuclei if model available
        if self.cls_model:
            # TODO: Implement nuclear patch extraction and classification
            pass
        
        # Generate visualization
        if save_dir:
            overlay = overlay_masks(image, pred_mask)
            cv2.imwrite(str(save_dir / 'segmentation.png'), 
                       cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Save analysis results
            self.nuclear_analyzer.generate_report(
                nuclei_features, 
                Path(image_path).stem,
                save_dir
            )
        
        return {
            'mask': pred_mask,
            'nuclei_features': nuclei_features,
            'density': density
        }
    
    def process_batch(self, 
                     image_paths: List[Union[str, Path]],
                     save_dir: Optional[Union[str, Path]] = None) -> List[Dict]:
        """
        Process multiple images
        
        Args:
            image_paths: List of paths to input images
            save_dir: Optional directory to save results
            
        Returns:
            List of dictionaries containing analysis results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.process_image(image_path, save_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                continue
        return results
