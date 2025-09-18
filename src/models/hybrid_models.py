"""
Hybrid CNN-LSTM Models for Multimodal Crop Health Analysis

Combines CNN feature extraction from hyperspectral imagery with
LSTM temporal modeling of sensor data for comprehensive crop monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import logging

from .cnn_models import CNNFeatureExtractor, ResNetFeatureExtractor, AttentionCNN
from .lstm_models import TemporalAttentionLSTM, MultiScaleLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CropHealthModel(nn.Module):
    """Main hybrid model for crop health assessment."""
    
    def __init__(self, 
                 n_bands: int = 200,
                 sensor_dim: int = 15,
                 cnn_hidden_dim: int = 256,
                 lstm_hidden_dim: int = 128,
                 n_classes: int = 3,
                 dropout: float = 0.3,
                 cnn_type: str = 'feature_extractor',
                 attention_type: str = 'additive'):
        """
        Initialize Crop Health Model.
        
        Args:
            n_bands: Number of hyperspectral bands
            sensor_dim: Dimension of sensor data features
            cnn_hidden_dim: CNN hidden dimension
            lstm_hidden_dim: LSTM hidden dimension
            n_classes: Number of health classes (healthy, stressed, diseased)
            dropout: Dropout rate
            cnn_type: Type of CNN ('feature_extractor', 'resnet', 'attention')
            attention_type: Type of attention for LSTM
        """
        super(CropHealthModel, self).__init__()
        
        self.n_bands = n_bands
        self.sensor_dim = sensor_dim
        self.cnn_hidden_dim = cnn_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.n_classes = n_classes
        
        # CNN for spatial-spectral feature extraction
        if cnn_type == 'feature_extractor':
            self.cnn_extractor = CNNFeatureExtractor(
                n_bands=n_bands,
                hidden_dim=cnn_hidden_dim,
                dropout=dropout
            )
        elif cnn_type == 'resnet':
            self.cnn_extractor = ResNetFeatureExtractor(
                n_bands=n_bands,
                hidden_dim=cnn_hidden_dim
            )
        elif cnn_type == 'attention':
            self.cnn_extractor = AttentionCNN(
                n_bands=n_bands,
                hidden_dim=cnn_hidden_dim,
                n_classes=cnn_hidden_dim,  # Use as feature extractor
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown CNN type: {cnn_type}")
        
        # Sensor data preprocessing
        self.sensor_processor = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64)
        )
        
        # LSTM for temporal modeling
        combined_input_dim = cnn_hidden_dim + 64  # CNN features + processed sensor features
        self.lstm_model = TemporalAttentionLSTM(
            input_dim=combined_input_dim,
            hidden_dim=lstm_hidden_dim,
            output_dim=n_classes,
            dropout=dropout,
            attention_type=attention_type
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_input_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, 
                img_sequence: torch.Tensor, 
                sensor_sequence: torch.Tensor,
                sequence_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass through the hybrid model.
        
        Args:
            img_sequence: Hyperspectral image sequence (batch, seq_len, height, width, bands)
            sensor_sequence: Sensor data sequence (batch, seq_len, sensor_dim)
            sequence_lengths: Actual sequence lengths for each batch item
        
        Returns:
            Predictions, attention weights, and intermediate features
        """
        batch_size, seq_len, height, width, n_bands = img_sequence.shape
        
        # Extract CNN features for each time step
        cnn_features = []
        for t in range(seq_len):
            img_t = img_sequence[:, t]  # (batch, height, width, bands)
            
            # Extract spatial-spectral features
            cnn_feat = self.cnn_extractor(img_t)  # (batch, cnn_hidden_dim)
            cnn_features.append(cnn_feat)
        
        # Stack CNN features across time
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch, seq_len, cnn_hidden_dim)
        
        # Process sensor data
        processed_sensor = self.sensor_processor(sensor_sequence)  # (batch, seq_len, 64)
        
        # Combine CNN and sensor features
        combined_features = torch.cat([cnn_features, processed_sensor], dim=2)
        
        # Apply fusion layer
        fused_features = self.fusion_layer(combined_features)
        
        # LSTM temporal modeling
        predictions, attention_weights = self.lstm_model(combined_features, sequence_lengths)
        
        # Prepare output dictionary with intermediate features
        intermediate_features = {
            'cnn_features': cnn_features,
            'sensor_features': processed_sensor,
            'combined_features': combined_features,
            'fused_features': fused_features
        }
        
        return predictions, attention_weights, intermediate_features


class MultiModalModel(nn.Module):
    """Advanced multimodal model with multiple fusion strategies."""
    
    def __init__(self,
                 n_bands: int = 200,
                 sensor_dim: int = 15,
                 n_classes: int = 3,
                 fusion_strategy: str = 'late',
                 dropout: float = 0.3):
        """
        Initialize MultiModal Model.
        
        Args:
            n_bands: Number of hyperspectral bands
            sensor_dim: Dimension of sensor data
            n_classes: Number of output classes
            fusion_strategy: Fusion strategy ('early', 'late', 'hybrid')
            dropout: Dropout rate
        """
        super(MultiModalModel, self).__init__()
        
        self.fusion_strategy = fusion_strategy
        self.n_classes = n_classes
        
        # Image processing branch
        self.image_cnn = CNNFeatureExtractor(
            n_bands=n_bands,
            hidden_dim=256,
            dropout=dropout
        )
        
        # Sensor processing branch
        self.sensor_lstm = TemporalAttentionLSTM(
            input_dim=sensor_dim,
            hidden_dim=128,
            output_dim=128,  # Feature extraction mode
            dropout=dropout
        )
        
        if fusion_strategy == 'early':
            # Early fusion: combine raw features before processing
            self.early_fusion = nn.Linear(n_bands + sensor_dim, n_bands + sensor_dim)
            self.unified_processor = CNNFeatureExtractor(
                n_bands=n_bands + sensor_dim,
                hidden_dim=256,
                dropout=dropout
            )
            self.classifier = nn.Linear(256, n_classes)
            
        elif fusion_strategy == 'late':
            # Late fusion: separate processing then combine predictions
            self.image_classifier = nn.Linear(256, n_classes)
            self.sensor_classifier = nn.Linear(128, n_classes)
            self.fusion_weights = nn.Parameter(torch.tensor([0.6, 0.4]))  # Learnable weights
            
        elif fusion_strategy == 'hybrid':
            # Hybrid fusion: multiple fusion points
            self.intermediate_fusion = nn.Sequential(
                nn.Linear(256 + 128, 256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.final_classifier = nn.Linear(256, n_classes)
            
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    def forward(self, 
                img_sequence: torch.Tensor,
                sensor_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multimodal fusion.
        
        Args:
            img_sequence: Image sequence (batch, seq_len, height, width, bands)
            sensor_sequence: Sensor sequence (batch, seq_len, sensor_dim)
        
        Returns:
            Dictionary with predictions and intermediate results
        """
        batch_size, seq_len = img_sequence.shape[:2]
        
        if self.fusion_strategy == 'early':
            # Early fusion approach
            # This is simplified - in practice, you'd need to align dimensions properly
            combined_features = torch.cat([
                img_sequence.mean(dim=(2, 3)),  # Spatial average of images
                sensor_sequence
            ], dim=-1)
            
            fused_features = self.early_fusion(combined_features)
            # Process through unified model (simplified)
            predictions = self.classifier(fused_features.mean(dim=1))
            
            return {'predictions': predictions}
        
        elif self.fusion_strategy == 'late':
            # Process images
            img_features = []
            for t in range(seq_len):
                img_feat = self.image_cnn(img_sequence[:, t])
                img_features.append(img_feat)
            img_features = torch.stack(img_features, dim=1).mean(dim=1)  # Temporal average
            
            # Process sensors
            sensor_features, _ = self.sensor_lstm(sensor_sequence)
            
            # Separate predictions
            img_predictions = self.image_classifier(img_features)
            sensor_predictions = self.sensor_classifier(sensor_features)
            
            # Weighted fusion of predictions
            fusion_weights = F.softmax(self.fusion_weights, dim=0)
            final_predictions = (fusion_weights[0] * img_predictions + 
                               fusion_weights[1] * sensor_predictions)
            
            return {
                'predictions': final_predictions,
                'image_predictions': img_predictions,
                'sensor_predictions': sensor_predictions,
                'fusion_weights': fusion_weights
            }
        
        elif self.fusion_strategy == 'hybrid':
            # Process both modalities
            img_features = []
            for t in range(seq_len):
                img_feat = self.image_cnn(img_sequence[:, t])
                img_features.append(img_feat)
            img_features = torch.stack(img_features, dim=1).mean(dim=1)
            
            sensor_features, attention_weights = self.sensor_lstm(sensor_sequence)
            
            # Intermediate fusion
            combined_features = torch.cat([img_features, sensor_features], dim=1)
            fused_features = self.intermediate_fusion(combined_features)
            
            # Final prediction
            final_predictions = self.final_classifier(fused_features)
            
            return {
                'predictions': final_predictions,
                'image_features': img_features,
                'sensor_features': sensor_features,
                'attention_weights': attention_weights
            }


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for robust predictions."""
    
    def __init__(self, 
                 models: List[nn.Module],
                 ensemble_method: str = 'average',
                 n_classes: int = 3):
        """
        Initialize Ensemble Model.
        
        Args:
            models: List of individual models
            ensemble_method: Method for combining predictions ('average', 'weighted', 'voting')
            n_classes: Number of output classes
        """
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.n_classes = n_classes
        
        if ensemble_method == 'weighted':
            # Learnable weights for each model
            self.model_weights = nn.Parameter(torch.ones(len(models)) / len(models))
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Returns:
            Dictionary with ensemble predictions and individual model outputs
        """
        individual_predictions = []
        
        # Get predictions from each model
        for model in self.models:
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    pred = model(*args, **kwargs)
                    if isinstance(pred, tuple):
                        pred = pred[0]  # Take first output if tuple
                    elif isinstance(pred, dict):
                        pred = pred['predictions']
                    individual_predictions.append(pred)
        
        individual_predictions = torch.stack(individual_predictions, dim=0)
        
        # Combine predictions
        if self.ensemble_method == 'average':
            ensemble_pred = torch.mean(individual_predictions, dim=0)
        
        elif self.ensemble_method == 'weighted':
            weights = F.softmax(self.model_weights, dim=0)
            ensemble_pred = torch.sum(weights.view(-1, 1, 1) * individual_predictions, dim=0)
        
        elif self.ensemble_method == 'voting':
            # Hard voting: take mode of predictions
            individual_classes = torch.argmax(individual_predictions, dim=-1)
            ensemble_classes = torch.mode(individual_classes, dim=0)[0]
            ensemble_pred = F.one_hot(ensemble_classes, num_classes=self.n_classes).float()
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return {
            'ensemble_predictions': ensemble_pred,
            'individual_predictions': individual_predictions
        }


class TransferLearningModel(nn.Module):
    """Model with transfer learning capabilities."""
    
    def __init__(self,
                 pretrained_model: nn.Module,
                 n_classes: int = 3,
                 freeze_features: bool = True):
        """
        Initialize Transfer Learning Model.
        
        Args:
            pretrained_model: Pre-trained base model
            n_classes: Number of target classes
            freeze_features: Whether to freeze feature extraction layers
        """
        super(TransferLearningModel, self).__init__()
        
        self.pretrained_model = pretrained_model
        self.n_classes = n_classes
        
        # Freeze feature extraction layers if specified
        if freeze_features:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        
        # Replace final classifier
        if hasattr(pretrained_model, 'classifier'):
            in_features = pretrained_model.classifier[-1].in_features
            self.pretrained_model.classifier[-1] = nn.Linear(in_features, n_classes)
        elif hasattr(pretrained_model, 'fc'):
            in_features = pretrained_model.fc.in_features
            self.pretrained_model.fc = nn.Linear(in_features, n_classes)
        else:
            # Add new classifier head
            self.new_classifier = nn.Linear(256, n_classes)  # Assume 256 features
    
    def forward(self, *args, **kwargs):
        """Forward pass through transfer learning model."""
        return self.pretrained_model(*args, **kwargs)
    
    def unfreeze_layers(self, num_layers: int = 1):
        """Unfreeze the last num_layers for fine-tuning."""
        layers = list(self.pretrained_model.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True


def create_hybrid_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create hybrid models.
    
    Args:
        model_type: Type of hybrid model ('crop_health', 'multimodal', 'ensemble')
        **kwargs: Model-specific parameters
    
    Returns:
        Initialized hybrid model
    """
    if model_type == 'crop_health':
        return CropHealthModel(**kwargs)
    elif model_type == 'multimodal':
        return MultiModalModel(**kwargs)
    elif model_type == 'ensemble':
        return EnsembleModel(**kwargs)
    elif model_type == 'transfer':
        return TransferLearningModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Test hybrid models with sample data."""
    logger.info("Testing hybrid models...")
    
    # Create sample data
    batch_size = 2
    seq_len = 12
    height, width = 32, 32
    n_bands = 200
    sensor_dim = 15
    
    sample_img_seq = torch.randn(batch_size, seq_len, height, width, n_bands)
    sample_sensor_seq = torch.randn(batch_size, seq_len, sensor_dim)
    sample_lengths = torch.tensor([10, 12])
    
    # Test CropHealthModel
    logger.info("Testing CropHealthModel...")
    crop_model = CropHealthModel(
        n_bands=n_bands,
        sensor_dim=sensor_dim,
        cnn_hidden_dim=256,
        lstm_hidden_dim=128,
        n_classes=3
    )
    
    try:
        with torch.no_grad():
            predictions, attention_weights, features = crop_model(
                sample_img_seq, sample_sensor_seq, sample_lengths
            )
            logger.info(f"CropHealthModel output shape: {predictions.shape}")
            logger.info(f"Attention weights shape: {attention_weights.shape}")
    except Exception as e:
        logger.error(f"Error testing CropHealthModel: {str(e)}")
    
    # Test MultiModalModel
    logger.info("Testing MultiModalModel...")
    multimodal_model = MultiModalModel(
        n_bands=n_bands,
        sensor_dim=sensor_dim,
        fusion_strategy='late'
    )
    
    try:
        with torch.no_grad():
            results = multimodal_model(sample_img_seq, sample_sensor_seq)
            logger.info(f"MultiModalModel predictions shape: {results['predictions'].shape}")
    except Exception as e:
        logger.error(f"Error testing MultiModalModel: {str(e)}")
    
    logger.info("Hybrid model testing completed")


if __name__ == "__main__":
    main()