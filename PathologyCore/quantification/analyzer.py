import numpy as np
from skimage import measure
from typing import List, Dict, Tuple
import pandas as pd
import json
from pathlib import Path

class NuclearAnalyzer:
    def __init__(self):
        """Initialize the nuclear analyzer"""
        pass

    def analyze_morphology(self, mask: np.ndarray) -> List[Dict]:
        """
        Analyze morphological features of nuclei in a binary mask.
        
        Args:
            mask (np.ndarray): Binary segmentation mask
            
        Returns:
            List[Dict]: List of dictionaries containing features for each nucleus
        """
        labeled_mask = measure.label(mask)
        properties = measure.regionprops(labeled_mask)
        
        nuclei_features = []
        for prop in properties:
            features = {
                'area': prop.area,
                'perimeter': prop.perimeter,
                'centroid_x': prop.centroid[1],
                'centroid_y': prop.centroid[0],
                'eccentricity': prop.eccentricity,
                'solidity': prop.solidity,
                'major_axis_length': prop.major_axis_length,
                'minor_axis_length': prop.minor_axis_length
            }
            nuclei_features.append(features)
        
        return nuclei_features

    def calculate_density(self, nuclei_features: List[Dict], image_size: Tuple[int, int]) -> float:
        """
        Calculate nuclear density (nuclei per unit area)
        
        Args:
            nuclei_features (List[Dict]): List of nuclei features
            image_size (Tuple[int, int]): Size of the image (height, width)
            
        Returns:
            float: Nuclear density
        """
        total_area = image_size[0] * image_size[1]
        num_nuclei = len(nuclei_features)
        return num_nuclei / total_area

    def generate_report(self, nuclei_features: List[Dict], image_id: str, save_dir: Path) -> Dict:
        """
        Generate analysis report and save to CSV/JSON
        
        Args:
            nuclei_features (List[Dict]): List of nuclei features
            image_id (str): Identifier for the image
            save_dir (Path): Directory to save reports
            
        Returns:
            Dict: Summary statistics
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate summary statistics
        summary = {
            'image_id': image_id,
            'num_nuclei': len(nuclei_features),
            'avg_area': np.mean([n['area'] for n in nuclei_features]),
            'avg_perimeter': np.mean([n['perimeter'] for n in nuclei_features]),
            'avg_eccentricity': np.mean([n['eccentricity'] for n in nuclei_features]),
            'avg_solidity': np.mean([n['solidity'] for n in nuclei_features])
        }
        
        # Save detailed features to CSV
        df = pd.DataFrame(nuclei_features)
        df['image_id'] = image_id
        df.to_csv(save_dir / f'{image_id}_detailed.csv', index=False)
        
        # Save summary to JSON
        with open(save_dir / f'{image_id}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
