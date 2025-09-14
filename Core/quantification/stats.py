import numpy as np
from typing import Dict, List
from pathlib import Path
import json

class Statistics:
    def __init__(self):
        """Initialize statistics calculator"""
        pass

    def calculate_basic_stats(self, measurements: List[float]) -> Dict[str, float]:
        """
        Calculate basic statistical measures
        
        Args:
            measurements (List[float]): List of measurements
            
        Returns:
            Dict[str, float]: Dictionary containing statistical measures
        """
        if not measurements:
            return {}
            
        stats = {
            'mean': np.mean(measurements),
            'std': np.std(measurements),
            'median': np.median(measurements),
            'min': np.min(measurements),
            'max': np.max(measurements),
            'count': len(measurements)
        }
        return stats

    def analyze_distribution(self, measurements: List[float], n_bins: int = 10) -> Dict:
        """
        Analyze the distribution of measurements
        
        Args:
            measurements (List[float]): List of measurements
            n_bins (int): Number of bins for histogram
            
        Returns:
            Dict: Distribution analysis results
        """
        hist, bin_edges = np.histogram(measurements, bins=n_bins)
        
        analysis = {
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'skewness': float(np.mean(((measurements - np.mean(measurements))/np.std(measurements))**3)),
            'kurtosis': float(np.mean(((measurements - np.mean(measurements))/np.std(measurements))**4))
        }
        return analysis

    def save_stats(self, stats: Dict, save_path: Path):
        """
        Save statistics to JSON file
        
        Args:
            stats (Dict): Statistics to save
            save_path (Path): Path to save the JSON file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def load_stats(self, load_path: Path) -> Dict:
        """
        Load statistics from JSON file
        
        Args:
            load_path (Path): Path to load the JSON file from
            
        Returns:
            Dict: Loaded statistics
        """
        with open(load_path, 'r') as f:
            return json.load(f)
