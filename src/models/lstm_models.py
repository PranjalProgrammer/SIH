"""
LSTM Models for Temporal Sequence Analysis

Implements LSTM architectures with attention mechanisms for processing
temporal sequences of crop health data and environmental sensor readings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalAttentionLSTM(nn.Module):
    """LSTM with attention mechanism for temporal modeling."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int, 
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True,
                 attention_type: str = 'additive'):
        """
        Initialize Temporal Attention LSTM.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            output_dim: Output dimension (number of classes)
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            attention_type: Type of attention ('additive', 'multiplicative', 'scaled_dot')
        """
        super(TemporalAttentionLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention_type = attention_type
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        if attention_type == 'additive':
            self.attention = AdditiveAttention(lstm_output_dim)
        elif attention_type == 'multiplicative':
            self.attention = MultiplicativeAttention(lstm_output_dim)
        elif attention_type == 'scaled_dot':
            self.attention = ScaledDotProductAttention(lstm_output_dim)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Sequence lengths for packed sequences
        
        Returns:
            Output predictions and attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        if lengths is not None:
            # Pack sequences for variable length inputs
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, (hidden, cell) = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True
            )
        else:
            lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        context_vector, attention_weights = self.attention(lstm_out, lengths)
        
        # Classification
        output = self.classifier(context_vector)
        
        return output, attention_weights


class AdditiveAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism."""
    
    def __init__(self, hidden_dim: int):
        super(AdditiveAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden_states: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply additive attention.
        
        Args:
            hidden_states: LSTM hidden states (batch_size, seq_len, hidden_dim)
            lengths: Sequence lengths
        
        Returns:
            Context vector and attention weights
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute attention scores
        # For additive attention: score = v^T * tanh(W * h_i + U * h_avg)
        h_avg = hidden_states.mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        h_avg_expanded = h_avg.expand(-1, seq_len, -1)   # (batch, seq_len, hidden_dim)
        
        scores = self.v(torch.tanh(self.W(hidden_states) + self.U(h_avg_expanded)))
        scores = scores.squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask for variable length sequences
        if lengths is not None:
            mask = self._create_mask(seq_len, lengths, hidden_states.device)
            scores = scores.masked_fill(mask, -float('inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=1)
        
        # Compute context vector
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * hidden_states, dim=1)
        
        return context_vector, attention_weights
    
    def _create_mask(self, seq_len: int, lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Create mask for variable length sequences."""
        batch_size = lengths.size(0)
        mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
        return mask


class MultiplicativeAttention(nn.Module):
    """Multiplicative (Luong) attention mechanism."""
    
    def __init__(self, hidden_dim: int):
        super(MultiplicativeAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
    def forward(self, hidden_states: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply multiplicative attention."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Use last hidden state as query
        query = hidden_states[:, -1, :].unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Compute attention scores: score = q^T * W * h_i
        transformed_hidden = self.W(hidden_states)  # (batch, seq_len, hidden_dim)
        scores = torch.bmm(query, transformed_hidden.transpose(1, 2))  # (batch, 1, seq_len)
        scores = scores.squeeze(1)  # (batch, seq_len)
        
        # Apply mask for variable length sequences
        if lengths is not None:
            mask = self._create_mask(seq_len, lengths, hidden_states.device)
            scores = scores.masked_fill(mask, -float('inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=1)
        
        # Compute context vector
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * hidden_states, dim=1)
        
        return context_vector, attention_weights
    
    def _create_mask(self, seq_len: int, lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Create mask for variable length sequences."""
        batch_size = lengths.size(0)
        mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
        return mask


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super(ScaledDotProductAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, hidden_states: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply scaled dot-product attention."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Generate queries, keys, and values
        Q = self.W_q(hidden_states)  # (batch, seq_len, hidden_dim)
        K = self.W_k(hidden_states)
        V = self.W_v(hidden_states)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask for variable length sequences
        if lengths is not None:
            mask = self._create_mask(seq_len, lengths, hidden_states.device)
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask, -float('inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )
        
        # Apply output projection
        output = self.W_o(attended_values)
        
        # Global average pooling to get context vector
        context_vector = output.mean(dim=1)
        
        # Average attention weights across heads for visualization
        avg_attention_weights = attention_weights.mean(dim=1).mean(dim=1)
        
        return context_vector, avg_attention_weights
    
    def _create_mask(self, seq_len: int, lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Create mask for variable length sequences."""
        batch_size = lengths.size(0)
        mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
        return mask


class MultiScaleLSTM(nn.Module):
    """Multi-scale LSTM for capturing different temporal patterns."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 scales: list = [1, 2, 4],
                 dropout: float = 0.3):
        """
        Initialize Multi-scale LSTM.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension for each scale
            output_dim: Output dimension
            scales: List of temporal scales (downsampling factors)
            dropout: Dropout rate
        """
        super(MultiScaleLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.scales = scales
        
        # Create LSTM for each scale
        self.lstm_layers = nn.ModuleList()
        for scale in scales:
            lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                dropout=dropout,
                bidirectional=True,
                batch_first=True
            )
            self.lstm_layers.append(lstm)
        
        # Feature fusion
        total_features = len(scales) * hidden_dim * 2  # *2 for bidirectional
        self.fusion = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale processing.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        
        Returns:
            Output predictions
        """
        batch_size, seq_len, input_dim = x.shape
        scale_features = []
        
        for i, (lstm, scale) in enumerate(zip(self.lstm_layers, self.scales)):
            # Downsample input for this scale
            if scale > 1:
                # Average pooling for downsampling
                x_downsampled = F.avg_pool1d(
                    x.transpose(1, 2), 
                    kernel_size=scale, 
                    stride=scale
                ).transpose(1, 2)
            else:
                x_downsampled = x
            
            # LSTM processing
            lstm_out, (hidden, cell) = lstm(x_downsampled)
            
            # Use last hidden state
            last_hidden = lstm_out[:, -1, :]
            scale_features.append(last_hidden)
        
        # Concatenate features from all scales
        combined_features = torch.cat(scale_features, dim=1)
        
        # Final prediction
        output = self.fusion(combined_features)
        
        return output


class GRUWithAttention(nn.Module):
    """GRU with attention mechanism as an alternative to LSTM."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        Initialize GRU with Attention.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            output_dim: Output dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional GRU
        """
        super(GRUWithAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = AdditiveAttention(gru_output_dim)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        
        # Apply attention
        context_vector, attention_weights = self.attention(gru_out)
        
        # Classification
        output = self.classifier(context_vector)
        
        return output, attention_weights


def create_lstm_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create LSTM models.
    
    Args:
        model_type: Type of LSTM model ('attention', 'multiscale', 'gru')
        **kwargs: Model-specific parameters
    
    Returns:
        Initialized LSTM model
    """
    if model_type == 'attention':
        return TemporalAttentionLSTM(**kwargs)
    elif model_type == 'multiscale':
        return MultiScaleLSTM(**kwargs)
    elif model_type == 'gru':
        return GRUWithAttention(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Test LSTM models with sample data."""
    logger.info("Testing LSTM models...")
    
    # Create sample data
    batch_size = 4
    seq_len = 24
    input_dim = 133  # CNN features + sensor features
    
    sample_data = torch.randn(batch_size, seq_len, input_dim)
    sample_lengths = torch.tensor([20, 24, 18, 22])
    
    # Test different LSTM models
    models = {
        'attention_additive': TemporalAttentionLSTM(
            input_dim=input_dim, 
            hidden_dim=128, 
            output_dim=3, 
            attention_type='additive'
        ),
        'attention_multiplicative': TemporalAttentionLSTM(
            input_dim=input_dim, 
            hidden_dim=128, 
            output_dim=3, 
            attention_type='multiplicative'
        ),
        'attention_scaled_dot': TemporalAttentionLSTM(
            input_dim=input_dim, 
            hidden_dim=128, 
            output_dim=3, 
            attention_type='scaled_dot'
        ),
        'multiscale': MultiScaleLSTM(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=3
        ),
        'gru_attention': GRUWithAttention(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=3
        )
    }
    
    for model_name, model in models.items():
        logger.info(f"Testing {model_name} model...")
        
        try:
            with torch.no_grad():
                if 'multiscale' in model_name:
                    output = model(sample_data)
                    attention_weights = None
                else:
                    output, attention_weights = model(sample_data, sample_lengths)
                
                logger.info(f"{model_name} output shape: {output.shape}")
                if attention_weights is not None:
                    logger.info(f"{model_name} attention weights shape: {attention_weights.shape}")
                
        except Exception as e:
            logger.error(f"Error testing {model_name}: {str(e)}")
    
    logger.info("LSTM model testing completed")


if __name__ == "__main__":
    main()