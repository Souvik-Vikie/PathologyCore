import pytest
import numpy as np
import torch
from pathlib import Path
from pathologycore.utils.metrics import dice_loss, dice_score, iou_score
from pathologycore.utils.visualization import calculate_metrics
from pathologycore.utils.dataset_utils import NuclearDataset
from pathologycore.quantification.analyzer import NuclearAnalyzer

@pytest.fixture
def sample_data():
    # Create sample prediction and target masks
    pred = torch.zeros((1, 1, 100, 100))
    target = torch.zeros((1, 1, 100, 100))
    
    # Add some shapes
    pred[0, 0, 20:40, 20:40] = 1
    target[0, 0, 25:45, 25:45] = 1
    
    return pred, target

def test_dice_loss(sample_data):
    pred, target = sample_data
    loss = dice_loss(pred, target)
    assert isinstance(loss, torch.Tensor)
    assert 0 <= loss.item() <= 1

def test_dice_score(sample_data):
    pred, target = sample_data
    score = dice_score(pred, target)
    assert isinstance(score, torch.Tensor)
    assert 0 <= score.item() <= 1

def test_iou_score(sample_data):
    pred, target = sample_data
    score = iou_score(pred, target)
    assert isinstance(score, torch.Tensor)
    assert 0 <= score.item() <= 1

def test_calculate_metrics(sample_data):
    pred, target = sample_data
    metrics = calculate_metrics(pred, target)
    
    assert isinstance(metrics, dict)
    assert all(0 <= v <= 1 for v in metrics.values())
    assert set(metrics.keys()) == {'dice', 'iou', 'precision', 'recall'}

def test_nuclear_analyzer():
    analyzer = NuclearAnalyzer()
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    # Add some nuclei
    mask[20:30, 20:30] = 1
    mask[60:70, 60:70] = 1
    
    features = analyzer.analyze_morphology(mask)
    assert len(features) == 2  # Should detect two nuclei
    
    density = analyzer.calculate_density(features, mask.shape)
    assert density > 0

def test_dataset():
    # Create temporary image and mask
    img_dir = Path('test_data/images')
    mask_dir = Path('test_data/masks')
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy image and mask
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    
    # Save test files
    img_path = img_dir / 'test.png'
    mask_path = mask_dir / 'test_mask.png'
    
    try:
        import cv2
        cv2.imwrite(str(img_path), img)
        cv2.imwrite(str(mask_path), mask)
        
        # Test dataset
        dataset = NuclearDataset(
            image_paths=[img_path],
            mask_paths=[mask_path],
            patch_size=(64, 64),
            augment=True
        )
        
        assert len(dataset) == 1
        image, mask = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.shape[0] == 3  # RGB channels
        assert mask.shape[0] == 1  # Single channel mask
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree('test_data')
