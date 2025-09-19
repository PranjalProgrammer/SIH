"""
Hybrid Models combining CNN and LSTM architectures
Multi-modal fusion for comprehensive agricultural analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_models import CropHealthCNN
from .lstm_models import SensorLSTM, BiLSTMEncoder
import logging

logger = logging.getLogger(__name__)

class CNNLSTMHybrid(nn.Module):
    """Hybrid CNN-LSTM model for spatio-temporal analysis"""
    
    def __init__(self, image_channels=3, sensor_features=7, num_classes=5, 
                 cnn_features=512, lstm_hidden=128):
        super(CNNLSTMHybrid, self).__init__()
        
        # CNN for spatial features
        self.cnn_backbone = CropHealthCNN(num_classes=cnn_features, input_channels=image_channels)
        # Remove the final classifier
        self.cnn_backbone.classifier = nn.Sequential(*list(self.cnn_backbone.classifier.children())[:-1])
        
        # LSTM for temporal sensor data
        self.sensor_lstm = SensorLSTM(
            input_size=sensor_features,
            hidden_size=lstm_hidden,
            output_size=lstm_hidden
        )
        
        # Attention mechanism for feature fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=cnn_features + lstm_hidden,
            num_heads=8,
            batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(cnn_features + lstm_hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images, sensor_sequences):
        # Extract CNN features from images
        batch_size = images.shape[0]
        cnn_features = self.cnn_backbone(images)  # (batch, cnn_features)
        
        # Extract LSTM features from sensor sequences
        lstm_features = self.sensor_lstm(sensor_sequences)  # (batch, lstm_hidden)
        
        # Combine features
        combined_features = torch.cat([cnn_features, lstm_features], dim=1)
        combined_features = combined_features.unsqueeze(1)  # Add sequence dimension
        
        # Apply cross-attention
        attended_features, _ = self.cross_attention(
            combined_features, combined_features, combined_features
        )
        attended_features = attended_features.squeeze(1)  # Remove sequence dimension
        
        # Final classification
        output = self.classifier(attended_features)
        return output

class MultiModalFusion(nn.Module):
    """Advanced multi-modal fusion for agricultural monitoring"""
    
    def __init__(self, hyperspectral_bands=200, rgb_channels=3, sensor_features=7, 
                 weather_features=10, num_classes=5):
        super(MultiModalFusion, self).__init__()
        
        # Hyperspectral image encoder
        self.hyperspectral_encoder = nn.Sequential(
            nn.Conv2d(hyperspectral_bands, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512)
        )
        
        # RGB image encoder
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(rgb_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256)
        )
        
        # Sensor data encoder
        self.sensor_encoder = BiLSTMEncoder(
            input_size=sensor_features,
            hidden_size=64,
            num_layers=2
        )
        
        # Weather data encoder
        self.weather_encoder = BiLSTMEncoder(
            input_size=weather_features,
            hidden_size=64,
            num_layers=2
        )
        
        # Feature fusion with attention
        total_features = 512 + 256 + 128 + 128  # hyperspectral + rgb + sensor + weather
        self.fusion_attention = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.Tanh(),
            nn.Linear(256, 4),  # 4 modalities
            nn.Softmax(dim=1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, hyperspectral_img, rgb_img, sensor_data, weather_data):
        # Encode each modality
        hyper_features = self.hyperspectral_encoder(hyperspectral_img)
        rgb_features = self.rgb_encoder(rgb_img)
        
        # Process temporal data
        sensor_out, _ = self.sensor_encoder(sensor_data)
        sensor_features = sensor_out[:, -1, :]  # Last time step
        
        weather_out, _ = self.weather_encoder(weather_data)
        weather_features = weather_out[:, -1, :]  # Last time step
        
        # Concatenate all features
        all_features = torch.cat([
            hyper_features, rgb_features, sensor_features, weather_features
        ], dim=1)
        
        # Compute attention weights for each modality
        attention_weights = self.fusion_attention(all_features)
        
        # Apply attention (simplified - in practice you'd weight each modality separately)
        weighted_features = all_features * attention_weights.mean(dim=1, keepdim=True)
        
        # Final classification
        output = self.classifier(weighted_features)
        return output, attention_weights

class TemporalCNN(nn.Module):
    """CNN for temporal sequence of images"""
    
    def __init__(self, sequence_length=7, image_channels=3, num_classes=5):
        super(TemporalCNN, self).__init__()
        
        self.sequence_length = sequence_length
        
        # Individual frame processing
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten()
        )
        
        # Temporal processing
        self.temporal_lstm = nn.LSTM(
            input_size=256 * 7 * 7,
            hidden_size=512,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image_sequence):
        # image_sequence shape: (batch, sequence, channels, height, width)
        batch_size, seq_len = image_sequence.shape[:2]
        
        # Process each frame
        frame_features = []
        for t in range(seq_len):
            frame = image_sequence[:, t]
            features = self.frame_encoder(frame)
            frame_features.append(features)
        
        # Stack temporal features
        temporal_features = torch.stack(frame_features, dim=1)
        
        # LSTM processing
        lstm_out, _ = self.temporal_lstm(temporal_features)
        
        # Use last time step for classification
        final_features = lstm_out[:, -1, :]
        output = self.classifier(final_features)
        
        return output

def create_hybrid_model(model_type='cnn_lstm', **kwargs):
    """Factory function to create hybrid models"""
    if model_type == 'cnn_lstm':
        return CNNLSTMHybrid(**kwargs)
    elif model_type == 'multimodal':
        return MultiModalFusion(**kwargs)
    elif model_type == 'temporal_cnn':
        return TemporalCNN(**kwargs)
    else:
        raise ValueError(f"Unknown hybrid model type: {model_type}")

def main():
    """Test hybrid model creation"""
    model = create_hybrid_model('cnn_lstm')
    logger.info(f"Created hybrid model with {sum(p.numel() for p in model.parameters())} parameters")

if __name__ == "__main__":
    main()