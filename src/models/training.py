"""
Model Training Module
Handles training of all agricultural AI models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class ModelTrainer:
    """General purpose model trainer"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    # Multi-modal inputs
                    inputs = [inp.to(self.device) for inp in batch[:-1]]
                    targets = batch[-1].to(self.device)
            else:
                inputs = batch.to(self.device)
                targets = inputs  # For unsupervised learning
            
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(inputs, list):
                outputs = self.model(*inputs)
            else:
                outputs = self.model(inputs)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take main output
            
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Handle different batch formats (same as training)
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                    else:
                        inputs = [inp.to(self.device) for inp in batch[:-1]]
                        targets = batch[-1].to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = inputs
                
                # Forward pass
                if isinstance(inputs, list):
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, num_epochs=100, learning_rate=0.001, 
              weight_decay=1e-4, patience=10, save_dir='/app/models'):
        """Full training loop"""
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        patience_counter = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate_epoch(val_loader, criterion)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.save_model(save_dir, f'best_model_epoch_{epoch+1}.pth')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training completed")
        return self.train_losses, self.val_losses
    
    def save_model(self, save_dir, filename):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Model loaded from {checkpoint_path}")

class AugmentedTrainer(ModelTrainer):
    """Trainer with data augmentation support"""
    
    def __init__(self, model, augmentation_transforms=None, **kwargs):
        super().__init__(model, **kwargs)
        self.augmentation_transforms = augmentation_transforms
    
    def apply_augmentation(self, inputs):
        """Apply data augmentation"""
        if self.augmentation_transforms and self.model.training:
            if isinstance(inputs, torch.Tensor):
                return self.augmentation_transforms(inputs)
            elif isinstance(inputs, list):
                return [self.augmentation_transforms(inp) if inp.dim() == 4 else inp for inp in inputs]
        return inputs

class MultiTaskTrainer(ModelTrainer):
    """Trainer for multi-task learning"""
    
    def __init__(self, model, task_weights=None, **kwargs):
        super().__init__(model, **kwargs)
        self.task_weights = task_weights or [1.0]  # Equal weights by default
    
    def compute_multitask_loss(self, outputs, targets, criterion):
        """Compute weighted multi-task loss"""
        if not isinstance(outputs, (list, tuple)):
            return criterion(outputs, targets)
        
        total_loss = 0
        for i, (output, target, weight) in enumerate(zip(outputs, targets, self.task_weights)):
            task_loss = criterion(output, target)
            total_loss += weight * task_loss
        
        return total_loss

def create_trainer(trainer_type='basic', model=None, **kwargs):
    """Factory function to create trainers"""
    if trainer_type == 'basic':
        return ModelTrainer(model, **kwargs)
    elif trainer_type == 'augmented':
        return AugmentedTrainer(model, **kwargs)
    elif trainer_type == 'multitask':
        return MultiTaskTrainer(model, **kwargs)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

def main():
    """Test trainer creation"""
    # This would normally use an actual model
    logger.info("Training module initialized")

if __name__ == "__main__":
    main()