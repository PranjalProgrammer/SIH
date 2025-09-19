"""
Model Evaluation Module
Comprehensive evaluation metrics and visualization
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
    
    def evaluate(self, test_loader, class_names=None):
        """Evaluate model on test set"""
        self.model.eval()
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Handle different batch formats
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
                    continue  # Skip if no targets
                
                # Forward pass
                if isinstance(inputs, list):
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                self.predictions.extend(predictions.cpu().numpy())
                self.true_labels.extend(targets.cpu().numpy())
                self.probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = self.calculate_metrics(class_names)
        
        return metrics
    
    def calculate_metrics(self, class_names=None):
        """Calculate comprehensive evaluation metrics"""
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.probabilities)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # AUC (for multi-class)
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except ValueError:
            auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'auc': auc,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_true, y_pred, target_names=class_names)
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        if auc:
            logger.info(f"AUC: {auc:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, metrics, class_names=None, save_path=None):
        """Plot confusion matrix"""
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, class_names=None, save_path=None):
        """Plot ROC curves for each class"""
        if len(np.unique(self.true_labels)) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(self.true_labels, 
                                   np.array(self.probabilities)[:, 1])
            auc = roc_auc_score(self.true_labels, 
                               np.array(self.probabilities)[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            
        else:
            # Multi-class classification
            n_classes = len(np.unique(self.true_labels))
            plt.figure(figsize=(10, 8))
            
            for i in range(n_classes):
                # One-vs-rest for each class
                y_true_binary = (np.array(self.true_labels) == i).astype(int)
                y_prob_class = np.array(self.probabilities)[:, i]
                
                fpr, tpr, _ = roc_curve(y_true_binary, y_prob_class)
                auc = roc_auc_score(y_true_binary, y_prob_class)
                
                class_name = class_names[i] if class_names else f'Class {i}'
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves - Multi-class')
            plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, train_losses, val_losses, save_path=None):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(np.gradient(train_losses), label='Training Loss Gradient')
        plt.plot(np.gradient(val_losses), label='Validation Loss Gradient')
        plt.title('Loss Gradients')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Gradient')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class CrossValidator:
    """K-fold cross validation"""
    
    def __init__(self, model_class, k_folds=5):
        self.model_class = model_class
        self.k_folds = k_folds
        self.fold_results = []
    
    def cross_validate(self, dataset, model_kwargs=None, train_kwargs=None):
        """Perform k-fold cross validation"""
        model_kwargs = model_kwargs or {}
        train_kwargs = train_kwargs or {}
        
        fold_size = len(dataset) // self.k_folds
        
        for fold in range(self.k_folds):
            logger.info(f"Starting fold {fold + 1}/{self.k_folds}")
            
            # Split data
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < self.k_folds - 1 else len(dataset)
            
            val_indices = list(range(start_idx, end_idx))
            train_indices = list(range(0, start_idx)) + list(range(end_idx, len(dataset)))
            
            # Create model and evaluator
            model = self.model_class(**model_kwargs)
            evaluator = ModelEvaluator(model)
            
            # This is a simplified version - in practice you'd need proper data loaders
            # val_loader = create_data_loader(dataset, val_indices)
            # metrics = evaluator.evaluate(val_loader)
            # self.fold_results.append(metrics)
        
        return self.fold_results
    
    def get_average_metrics(self):
        """Get average metrics across all folds"""
        if not self.fold_results:
            return None
        
        avg_metrics = {}
        for key in self.fold_results[0].keys():
            if key not in ['confusion_matrix', 'classification_report']:
                values = [fold[key] for fold in self.fold_results if fold[key] is not None]
                if values:
                    avg_metrics[key] = np.mean(values)
                    avg_metrics[f'{key}_std'] = np.std(values)
        
        return avg_metrics

def main():
    """Test evaluation functions"""
    logger.info("Evaluation module initialized")

if __name__ == "__main__":
    main()