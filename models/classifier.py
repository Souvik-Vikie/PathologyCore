import torch
import torch.nn as nn
import torchvision.models as models

class NuclearClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        """
        Nuclear classifier based on ResNet50 backbone
        Args:
            num_classes (int): Number of nuclear classes to predict
            pretrained (bool): Whether to use pretrained weights
        """
        super().__init__()
        
        # Load ResNet50 backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Apply attention
        attention = self.attention(features)
        features = features * attention
        
        # Global average pooling
        features = torch.mean(features, dim=[2, 3])
        
        # Classification
        output = self.classifier(features)
        return output

    def predict_proba(self, x):
        """
        Get probability predictions
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Class probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def extract_features(self, x):
        """
        Extract features from the backbone
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            torch.Tensor: Feature vectors
        """
        with torch.no_grad():
            features = self.features(x)
            features = torch.mean(features, dim=[2, 3])
            return features
