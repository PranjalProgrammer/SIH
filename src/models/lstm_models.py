"""
LSTM Models for Time Series Analysis
Long Short-Term Memory networks for temporal agricultural data
"""

import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SensorLSTM(nn.Module):
    """LSTM for sensor time series prediction"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(SensorLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last time step
        last_output = attn_out[:, -1, :]
        
        # Classify
        output = self.classifier(last_output)
        return output

class CropGrowthLSTM(nn.Module):
    """LSTM for crop growth stage prediction"""
    
    def __init__(self, sensor_features=7, image_features=512, hidden_size=256, num_classes=5):
        super(CropGrowthLSTM, self).__init__()
        
        # Sensor data processing
        self.sensor_lstm = nn.LSTM(
            input_size=sensor_features,
            hidden_size=hidden_size // 2,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # Image feature processing
        self.image_encoder = nn.Sequential(
            nn.Linear(image_features, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Combined processing
        self.fusion_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, sensor_data, image_features):
        # Process sensor data
        sensor_out, _ = self.sensor_lstm(sensor_data)
        
        # Process image features
        batch_size, seq_len = sensor_data.shape[:2]
        image_out = self.image_encoder(image_features)
        image_out = image_out.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Fusion
        combined = torch.cat([sensor_out, image_out], dim=-1)
        fusion_out, _ = self.fusion_lstm(combined)
        
        # Classification
        output = self.classifier(fusion_out[:, -1, :])
        return output

class WeatherLSTM(nn.Module):
    """LSTM for weather-based crop risk assessment"""
    
    def __init__(self, weather_features=10, hidden_size=128, sequence_length=30):
        super(WeatherLSTM, self).__init__()
        
        self.sequence_length = sequence_length
        
        # Weather pattern encoder
        self.weather_lstm = nn.LSTM(
            input_size=weather_features,
            hidden_size=hidden_size,
            num_layers=3,
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Risk prediction
        self.risk_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)  # [drought, flood, pest, disease] risks
        )
    
    def forward(self, weather_data):
        # LSTM processing
        lstm_out, _ = self.weather_lstm(weather_data)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)
        
        # Weighted sum
        context = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Risk prediction
        risks = self.risk_predictor(context)
        
        return risks, attention_weights

class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder for sequence processing"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(BiLSTMEncoder, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
    
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        return lstm_out, (hidden, cell)

def create_lstm_model(model_type='sensor', **kwargs):
    """Factory function to create LSTM models"""
    if model_type == 'sensor':
        return SensorLSTM(**kwargs)
    elif model_type == 'crop_growth':
        return CropGrowthLSTM(**kwargs)
    elif model_type == 'weather':
        return WeatherLSTM(**kwargs)
    elif model_type == 'encoder':
        return BiLSTMEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown LSTM model type: {model_type}")

def main():
    """Test LSTM model creation"""
    model = create_lstm_model('sensor', input_size=7, hidden_size=128)
    logger.info(f"Created LSTM model with {sum(p.numel() for p in model.parameters())} parameters")

if __name__ == "__main__":
    main()