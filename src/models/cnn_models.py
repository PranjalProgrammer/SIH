"""
CNN Models for Agricultural Image Analysis
Convolutional Neural Networks for crop health assessment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

class CropHealthCNN(nn.Module):
    """CNN for crop health classification"""
    
    def __init__(self, num_classes=5, input_channels=3):
        super(CropHealthCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class HyperspectralCNN(nn.Module):
    """CNN for hyperspectral image analysis"""
    
    def __init__(self, num_bands=200, num_classes=5):
        super(HyperspectralCNN, self).__init__()
        
        # Spectral feature extraction
        self.spectral_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(32)
        )
        
        # Spatial feature extraction
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(num_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(128 * 32 + 128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, bands, height, width)
        batch_size, bands, height, width = x.shape
        
        # Spectral processing
        spectral_features = x.mean(dim=(2, 3))  # Global average pooling
        spectral_features = spectral_features.unsqueeze(1)  # Add channel dim
        spectral_out = self.spectral_conv(spectral_features)
        spectral_out = spectral_out.flatten(1)
        
        # Spatial processing
        spatial_out = self.spatial_conv(x)
        spatial_out = spatial_out.flatten(1)
        
        # Fusion
        combined = torch.cat([spectral_out, spatial_out], dim=1)
        output = self.classifier(combined)
        
        return output

class ResNetFeatureExtractor(nn.Module):
    """ResNet-based feature extractor for transfer learning"""
    
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNetFeatureExtractor, self).__init__()
        
        # Load pretrained ResNet
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Modify first layer for different input channels if needed
        # self.backbone.conv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def create_model(model_type='cnn', **kwargs):
    """Factory function to create models"""
    if model_type == 'cnn':
        return CropHealthCNN(**kwargs)
    elif model_type == 'hyperspectral':
        return HyperspectralCNN(**kwargs)
    elif model_type == 'resnet':
        return ResNetFeatureExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    """Test model creation"""
    model = create_model('cnn', num_classes=5)
    logger.info(f"Created CNN model with {sum(p.numel() for p in model.parameters())} parameters")

if __name__ == "__main__":
    main()