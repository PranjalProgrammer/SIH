"""
CNN Models for Spatial and Spectral Feature Extraction

Implements various CNN architectures for processing hyperspectral imagery
and extracting spatial-spectral features for crop health analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpectralCNN(nn.Module):
    """CNN for extracting spectral features from hyperspectral data."""
    
    def __init__(self, 
                 n_bands: int = 200,
                 n_classes: int = 3,
                 dropout: float = 0.3):
        """
        Initialize Spectral CNN.
        
        Args:
            n_bands: Number of spectral bands
            n_classes: Number of output classes
            dropout: Dropout rate
        """
        super(SpectralCNN, self).__init__()
        
        self.n_bands = n_bands
        self.n_classes = n_classes
        
        # 1D Convolutional layers for spectral processing
        self.spectral_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, n_bands) for pixel-wise classification
               or (batch_size, height, width, n_bands) for full image
        
        Returns:
            Output tensor with class predictions
        """
        if x.dim() == 4:  # Full image input
            batch_size, height, width, n_bands = x.shape
            # Reshape to process each pixel independently
            x = x.view(-1, n_bands)
            
        # Add channel dimension for 1D conv
        x = x.unsqueeze(1)  # (batch_size * pixels, 1, n_bands)
        
        # Extract spectral features
        features = self.spectral_conv(x)
        features = features.squeeze(-1)  # Remove last dimension
        
        # Classify
        output = self.classifier(features)
        
        return output


class SpatialCNN(nn.Module):
    """CNN for extracting spatial features from imagery."""
    
    def __init__(self, 
                 in_channels: int = 10,
                 n_classes: int = 3,
                 dropout: float = 0.3):
        """
        Initialize Spatial CNN.
        
        Args:
            in_channels: Number of input channels (selected spectral bands or features)
            n_classes: Number of output classes
            dropout: Dropout rate
        """
        super(SpatialCNN, self).__init__()
        
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        # 2D Convolutional layers for spatial processing
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output tensor with class predictions
        """
        # Extract spatial features
        features = self.spatial_conv(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Classify
        output = self.classifier(features)
        
        return output


class CNNFeatureExtractor(nn.Module):
    """Enhanced CNN for extracting both spatial and spectral features."""
    
    def __init__(self, 
                 n_bands: int = 200,
                 patch_size: int = 32,
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        """
        Initialize CNN Feature Extractor.
        
        Args:
            n_bands: Number of spectral bands
            patch_size: Size of spatial patches
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
        """
        super(CNNFeatureExtractor, self).__init__()
        
        self.n_bands = n_bands
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # Spectral feature extraction branch
        self.spectral_branch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Spatial feature extraction branch
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(n_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(512, hidden_dim),  # 256 + 256 from both branches
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features.
        
        Args:
            x: Input tensor of shape (batch_size, height, width, n_bands)
        
        Returns:
            Extracted features of shape (batch_size, hidden_dim)
        """
        batch_size, height, width, n_bands = x.shape
        
        # Spectral features - process center pixel spectrum
        center_h, center_w = height // 2, width // 2
        center_spectrum = x[:, center_h, center_w, :].unsqueeze(1)  # (batch, 1, n_bands)
        spectral_features = self.spectral_branch(center_spectrum)
        spectral_features = spectral_features.squeeze(-1)  # (batch, 256)
        
        # Spatial features - transpose for conv2d format
        x_spatial = x.permute(0, 3, 1, 2)  # (batch, n_bands, height, width)
        spatial_features = self.spatial_branch(x_spatial)
        spatial_features = spatial_features.view(batch_size, -1)  # (batch, 256)
        
        # Fuse features
        combined_features = torch.cat([spectral_features, spatial_features], dim=1)
        fused_features = self.fusion(combined_features)
        
        return fused_features


class ResidualBlock(nn.Module):
    """Residual block for deeper CNN architectures."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out


class ResNetFeatureExtractor(nn.Module):
    """ResNet-based feature extractor for hyperspectral imagery."""
    
    def __init__(self, 
                 n_bands: int = 200,
                 num_blocks: list = [2, 2, 2, 2],
                 hidden_dim: int = 512):
        """
        Initialize ResNet Feature Extractor.
        
        Args:
            n_bands: Number of spectral bands
            num_blocks: Number of residual blocks in each layer
            hidden_dim: Output feature dimension
        """
        super(ResNetFeatureExtractor, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(n_bands, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        # Global average pooling and final linear layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, hidden_dim)
        
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a layer with residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, n_bands, height, width)
        
        Returns:
            Feature tensor (batch, hidden_dim)
        """
        # If input is (batch, height, width, n_bands), transpose
        if x.dim() == 4 and x.shape[-1] < x.shape[1]:
            x = x.permute(0, 3, 1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class AttentionBlock(nn.Module):
    """Attention mechanism for CNN feature maps."""
    
    def __init__(self, in_channels: int):
        super(AttentionBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to feature maps.
        
        Args:
            x: Input feature maps (batch, channels, height, width)
        
        Returns:
            Attention-weighted feature maps
        """
        attention_map = self.sigmoid(self.conv(x))
        return x * attention_map


class AttentionCNN(nn.Module):
    """CNN with attention mechanism for hyperspectral analysis."""
    
    def __init__(self, 
                 n_bands: int = 200,
                 n_classes: int = 3,
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        """
        Initialize Attention CNN.
        
        Args:
            n_bands: Number of spectral bands
            n_classes: Number of output classes
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
        """
        super(AttentionCNN, self).__init__()
        
        # Feature extraction layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            AttentionBlock(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            AttentionBlock(128),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            AttentionBlock(256),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Ensure correct input format
        if x.dim() == 4 and x.shape[-1] < x.shape[1]:
            x = x.permute(0, 3, 1, 2)
        
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        
        return output


def create_cnn_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create CNN models.
    
    Args:
        model_type: Type of CNN model ('spectral', 'spatial', 'feature_extractor', 
                   'resnet', 'attention')
        **kwargs: Model-specific parameters
    
    Returns:
        Initialized CNN model
    """
    if model_type == 'spectral':
        return SpectralCNN(**kwargs)
    elif model_type == 'spatial':
        return SpatialCNN(**kwargs)
    elif model_type == 'feature_extractor':
        return CNNFeatureExtractor(**kwargs)
    elif model_type == 'resnet':
        return ResNetFeatureExtractor(**kwargs)
    elif model_type == 'attention':
        return AttentionCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Test CNN models with sample data."""
    logger.info("Testing CNN models...")
    
    # Create sample data
    batch_size = 4
    height, width = 32, 32
    n_bands = 200
    
    # Sample hyperspectral data
    sample_data = torch.randn(batch_size, height, width, n_bands)
    
    # Test different CNN models
    models = {
        'spectral': SpectralCNN(n_bands=n_bands),
        'spatial': SpatialCNN(in_channels=10),
        'feature_extractor': CNNFeatureExtractor(n_bands=n_bands),
        'resnet': ResNetFeatureExtractor(n_bands=n_bands),
        'attention': AttentionCNN(n_bands=n_bands)
    }
    
    for model_name, model in models.items():
        logger.info(f"Testing {model_name} model...")
        
        try:
            if model_name == 'spatial':
                # Use first 10 bands for spatial model
                input_data = sample_data[:, :, :, :10].permute(0, 3, 1, 2)
            else:
                input_data = sample_data
            
            with torch.no_grad():
                output = model(input_data)
                logger.info(f"{model_name} output shape: {output.shape}")
                
        except Exception as e:
            logger.error(f"Error testing {model_name}: {str(e)}")
    
    logger.info("CNN model testing completed")


if __name__ == "__main__":
    main()