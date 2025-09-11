# PathologyCore

A comprehensive deep learning toolkit for digital pathology image analysis, focusing on nuclear segmentation, classification, and quantification.

## Features

- **Nuclear Segmentation**: U-Net based architecture for precise nuclear segmentation
- **Cell Classification**: Multi-class nuclear type classification
- **Quantification**: Morphological analysis and statistical reporting
- **Model Optimization**: Pruning and quantization for efficient deployment

## Getting Started

### Prerequisites

```bash
python>=3.8
cuda>=11.0 (optional, for GPU support)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Souvik-Vikie/PathologyCore.git
cd PathologyCore
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Training Segmentation Model

```python
from pathologycore.segmentation import UNet, SegmentationTrainer
from pathologycore.utils import NuclearDataset

# Create model and dataset
model = UNet(n_channels=3, n_classes=1)
dataset = NuclearDataset(image_dir='data/train')

# Train model
trainer = SegmentationTrainer(model)
trainer.train(dataset)
```

### Running Inference

```python
from pathologycore.core import NuclearAnalysisPipeline

# Create pipeline
pipeline = NuclearAnalysisPipeline(model_path='models/unet_best.pth')

# Process image
results = pipeline.process_image('path/to/image.png')
```

## Project Structure

```
PathologyCore/
├── configs/               # Configuration files
├── data/                 # Dataset storage
│   ├── processed/       
│   └── raw/             
├── models/               # Model implementations
├── notebooks/            # Jupyter notebooks
├── pathologycore/        # Main package
│   ├── core.py
│   ├── segmentation/
│   ├── classification/
│   └── utils/
├── results/              # Experiment outputs
└── tests/               # Unit tests
```

##  Configuration

Configuration files in `configs/` control model architecture, training parameters, and analysis settings:

- `seg_config.yaml`: Segmentation model configuration
- `cls_config.yaml`: Classification model configuration
- `quant_config.yaml`: Quantification pipeline settings

## Results

Results are saved in the `results/` directory:
- Model checkpoints
- Training metrics
- Validation visualizations
- Quantification reports

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

##  Acknowledgments

- PanNuke Dataset
- PyTorch Team
- OpenCV Community
