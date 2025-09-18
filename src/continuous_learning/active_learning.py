"""
Active Learning System for Precision Agriculture

Implements intelligent sample selection strategies to improve model
performance with minimal annotation effort.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import random
import heapq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UncertaintySampler:
    """Sample selection based on prediction uncertainty."""
    
    def __init__(self, 
                 method: str = 'entropy',
                 temperature: float = 1.0):
        """
        Initialize Uncertainty Sampler.
        
        Args:
            method: Uncertainty method ('entropy', 'margin', 'least_confident')
            temperature: Temperature scaling for uncertainty calibration
        """
        self.method = method
        self.temperature = temperature
    
    def calculate_uncertainty(self, 
                            predictions: torch.Tensor,
                            method: Optional[str] = None) -> np.ndarray:
        """
        Calculate uncertainty scores for predictions.
        
        Args:
            predictions: Model predictions (batch_size, n_classes)
            method: Override default uncertainty method
            
        Returns:
            Uncertainty scores for each sample
        """
        method = method or self.method
        
        # Apply temperature scaling
        scaled_predictions = predictions / self.temperature
        probabilities = torch.softmax(scaled_predictions, dim=1)
        
        if method == 'entropy':
            # Entropy-based uncertainty
            log_probs = torch.log_softmax(scaled_predictions, dim=1)
            entropy = -torch.sum(probabilities * log_probs, dim=1)
            return entropy.cpu().numpy()
        
        elif method == 'margin':
            # Margin-based uncertainty (difference between top 2 predictions)
            sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            return (1 - margin).cpu().numpy()  # Higher uncertainty = lower margin
        
        elif method == 'least_confident':
            # Least confident uncertainty
            max_probs, _ = torch.max(probabilities, dim=1)
            return (1 - max_probs).cpu().numpy()
        
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
    
    def select_samples(self, 
                      predictions: torch.Tensor,
                      n_samples: int,
                      existing_indices: Optional[List[int]] = None) -> List[int]:
        """
        Select samples based on uncertainty.
        
        Args:
            predictions: Model predictions
            n_samples: Number of samples to select
            existing_indices: Indices of already labeled samples
            
        Returns:
            Indices of selected samples
        """
        uncertainty_scores = self.calculate_uncertainty(predictions)
        
        # Exclude existing indices
        if existing_indices is not None:
            uncertainty_scores[existing_indices] = -np.inf
        
        # Select top uncertain samples
        selected_indices = np.argsort(uncertainty_scores)[-n_samples:].tolist()
        
        return selected_indices


class DiversitySampler:
    """Sample selection based on feature diversity."""
    
    def __init__(self, 
                 method: str = 'kmeans',
                 feature_dim: Optional[int] = None):
        """
        Initialize Diversity Sampler.
        
        Args:
            method: Diversity method ('kmeans', 'core_set', 'random')
            feature_dim: Feature dimension for PCA reduction
        """
        self.method = method
        self.feature_dim = feature_dim
    
    def select_samples(self, 
                      features: np.ndarray,
                      n_samples: int,
                      existing_indices: Optional[List[int]] = None) -> List[int]:
        """
        Select diverse samples based on feature representations.
        
        Args:
            features: Feature representations (n_samples, feature_dim)
            n_samples: Number of samples to select
            existing_indices: Indices of already labeled samples
            
        Returns:
            Indices of selected samples
        """
        # Reduce dimensionality if needed
        if self.feature_dim and features.shape[1] > self.feature_dim:
            pca = PCA(n_components=self.feature_dim)
            features = pca.fit_transform(features)
        
        # Create mask for available samples
        available_mask = np.ones(len(features), dtype=bool)
        if existing_indices is not None:
            available_mask[existing_indices] = False
        
        available_features = features[available_mask]
        available_indices = np.where(available_mask)[0]
        
        if self.method == 'kmeans':
            return self._kmeans_selection(available_features, available_indices, n_samples)
        elif self.method == 'core_set':
            return self._core_set_selection(available_features, available_indices, n_samples)
        elif self.method == 'random':
            return self._random_selection(available_indices, n_samples)
        else:
            raise ValueError(f"Unknown diversity method: {self.method}")
    
    def _kmeans_selection(self, 
                         features: np.ndarray, 
                         indices: np.ndarray, 
                         n_samples: int) -> List[int]:
        """Select samples using K-means clustering."""
        if len(features) <= n_samples:
            return indices.tolist()
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Select sample closest to each cluster center
        selected_indices = []
        for i in range(n_samples):
            cluster_mask = cluster_labels == i
            if not np.any(cluster_mask):
                continue
            
            cluster_features = features[cluster_mask]
            cluster_indices = indices[cluster_mask]
            
            # Find closest sample to cluster center
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_features - center, axis=1)
            closest_idx = np.argmin(distances)
            
            selected_indices.append(cluster_indices[closest_idx])
        
        return selected_indices
    
    def _core_set_selection(self, 
                           features: np.ndarray, 
                           indices: np.ndarray, 
                           n_samples: int) -> List[int]:
        """Select samples using core-set approach."""
        if len(features) <= n_samples:
            return indices.tolist()
        
        selected_indices = []
        remaining_indices = list(range(len(features)))
        
        # Start with random sample
        first_idx = random.choice(remaining_indices)
        selected_indices.append(indices[first_idx])
        remaining_indices.remove(first_idx)
        
        # Iteratively select samples that are farthest from selected set
        for _ in range(n_samples - 1):
            if not remaining_indices:
                break
            
            max_min_distance = -1
            best_idx = None
            
            for idx in remaining_indices:
                # Calculate minimum distance to selected samples
                distances = []
                for sel_idx in selected_indices:
                    sel_feature_idx = np.where(indices == sel_idx)[0][0]
                    dist = np.linalg.norm(features[idx] - features[sel_feature_idx])
                    distances.append(dist)
                
                min_distance = min(distances) if distances else 0
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(indices[best_idx])
                remaining_indices.remove(best_idx)
        
        return selected_indices
    
    def _random_selection(self, indices: np.ndarray, n_samples: int) -> List[int]:
        """Random sample selection."""
        n_samples = min(n_samples, len(indices))
        return random.sample(indices.tolist(), n_samples)


class ActiveLearner:
    """Main active learning coordinator."""
    
    def __init__(self, 
                 model: nn.Module,
                 uncertainty_sampler: Optional[UncertaintySampler] = None,
                 diversity_sampler: Optional[DiversitySampler] = None,
                 strategy: str = 'uncertainty',
                 combination_weight: float = 0.5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Active Learner.
        
        Args:
            model: Model for making predictions
            uncertainty_sampler: Uncertainty-based sampler
            diversity_sampler: Diversity-based sampler
            strategy: Selection strategy ('uncertainty', 'diversity', 'combined')
            combination_weight: Weight for combining uncertainty and diversity
            device: Computation device
        """
        self.model = model.to(device)
        self.device = device
        self.strategy = strategy
        self.combination_weight = combination_weight
        
        # Initialize samplers
        self.uncertainty_sampler = uncertainty_sampler or UncertaintySampler()
        self.diversity_sampler = diversity_sampler or DiversitySampler()
        
        # Tracking
        self.labeled_indices = set()
        self.selection_history = []
        
    def select_samples_for_annotation(self, 
                                    unlabeled_data: torch.utils.data.Dataset,
                                    n_samples: int,
                                    batch_size: int = 32) -> List[int]:
        """
        Select samples for annotation using active learning.
        
        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            batch_size: Batch size for model inference
            
        Returns:
            Indices of selected samples
        """
        logger.info(f"Selecting {n_samples} samples using {self.strategy} strategy")
        
        # Get model predictions and features for unlabeled data
        predictions, features = self._get_model_outputs(unlabeled_data, batch_size)
        
        # Select samples based on strategy
        if self.strategy == 'uncertainty':
            selected_indices = self.uncertainty_sampler.select_samples(
                predictions, n_samples, list(self.labeled_indices)
            )
        
        elif self.strategy == 'diversity':
            selected_indices = self.diversity_sampler.select_samples(
                features, n_samples, list(self.labeled_indices)
            )
        
        elif self.strategy == 'combined':
            selected_indices = self._combined_selection(
                predictions, features, n_samples
            )
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Update labeled indices
        self.labeled_indices.update(selected_indices)
        
        # Record selection
        self.selection_history.append({
            'iteration': len(self.selection_history),
            'strategy': self.strategy,
            'selected_indices': selected_indices,
            'n_samples': len(selected_indices)
        })
        
        logger.info(f"Selected {len(selected_indices)} samples for annotation")
        return selected_indices
    
    def _get_model_outputs(self, 
                          dataset: torch.utils.data.Dataset,
                          batch_size: int) -> Tuple[torch.Tensor, np.ndarray]:
        """Get model predictions and features for dataset."""
        self.model.eval()
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        
        all_predictions = []
        all_features = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    inputs = self._extract_inputs_from_batch(batch)
                else:
                    inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                
                # Move to device
                if isinstance(inputs, tuple):
                    inputs = tuple(inp.to(self.device) for inp in inputs)
                else:
                    inputs = inputs.to(self.device)
                
                # Forward pass
                try:
                    if isinstance(inputs, tuple):
                        outputs = self.model(*inputs)
                    else:
                        outputs = self.model(inputs)
                    
                    # Extract predictions and features
                    if isinstance(outputs, tuple):
                        predictions = outputs[0]
                        # Try to get intermediate features
                        features = self._extract_features(inputs)
                    elif isinstance(outputs, dict):
                        predictions = outputs['predictions']
                        features = outputs.get('features', self._extract_features(inputs))
                    else:
                        predictions = outputs
                        features = self._extract_features(inputs)
                    
                    all_predictions.append(predictions.cpu())
                    all_features.append(features)
                    
                except Exception as e:
                    logger.warning(f"Failed to process batch: {e}")
                    continue
        
        # Concatenate results
        predictions_tensor = torch.cat(all_predictions, dim=0)
        features_array = np.vstack(all_features)
        
        return predictions_tensor, features_array
    
    def _extract_inputs_from_batch(self, batch: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Extract model inputs from batch dictionary."""
        if 'image_sequence' in batch and 'sensor_sequence' in batch:
            return (batch['image_sequence'], batch['sensor_sequence'])
        elif 'input' in batch:
            return batch['input']
        else:
            # Try to find tensor values
            tensor_values = [v for v in batch.values() if isinstance(v, torch.Tensor)]
            return tensor_values[0] if len(tensor_values) == 1 else tuple(tensor_values)
    
    def _extract_features(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> np.ndarray:
        """Extract features from model inputs (simplified)."""
        # This is a simplified feature extraction
        # In practice, you might want to use intermediate layer outputs
        if isinstance(inputs, tuple):
            # For multi-modal inputs, concatenate flattened features
            features = []
            for inp in inputs:
                # Take spatial and temporal averages, then flatten
                if inp.dim() == 5:  # (batch, seq, h, w, c)
                    feat = inp.mean(dim=(1, 2, 3))  # Average over seq, h, w
                elif inp.dim() == 4:  # (batch, h, w, c) or (batch, seq, features)
                    feat = inp.mean(dim=(1, 2)) if inp.size(2) > inp.size(3) else inp.mean(dim=1)
                elif inp.dim() == 3:  # (batch, seq, features)
                    feat = inp.mean(dim=1)
                else:
                    feat = inp
                
                features.append(feat.flatten(1))
            
            combined_features = torch.cat(features, dim=1)
        else:
            # Single input
            if inputs.dim() > 2:
                combined_features = inputs.flatten(1)
            else:
                combined_features = inputs
        
        return combined_features.cpu().numpy()
    
    def _combined_selection(self, 
                           predictions: torch.Tensor,
                           features: np.ndarray,
                           n_samples: int) -> List[int]:
        """Combine uncertainty and diversity for sample selection."""
        # Get uncertainty scores
        uncertainty_scores = self.uncertainty_sampler.calculate_uncertainty(predictions)
        
        # Normalize uncertainty scores
        uncertainty_scores = (uncertainty_scores - uncertainty_scores.min()) / (
            uncertainty_scores.max() - uncertainty_scores.min() + 1e-8
        )
        
        # Get diversity-based candidates (more than needed)
        diversity_candidates = self.diversity_sampler.select_samples(
            features, min(n_samples * 3, len(features)), list(self.labeled_indices)
        )
        
        # Score combination for candidates
        combined_scores = []
        for idx in diversity_candidates:
            uncertainty_score = uncertainty_scores[idx]
            diversity_score = 1.0  # All candidates have diversity
            
            combined_score = (self.combination_weight * uncertainty_score + 
                            (1 - self.combination_weight) * diversity_score)
            
            combined_scores.append((combined_score, idx))
        
        # Select top scoring samples
        combined_scores.sort(reverse=True)
        selected_indices = [idx for _, idx in combined_scores[:n_samples]]
        
        return selected_indices
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics about active learning selections."""
        if not self.selection_history:
            return {}
        
        total_selected = sum(entry['n_samples'] for entry in self.selection_history)
        
        stats = {
            'total_iterations': len(self.selection_history),
            'total_samples_selected': total_selected,
            'average_samples_per_iteration': total_selected / len(self.selection_history),
            'current_labeled_count': len(self.labeled_indices),
            'selection_history': self.selection_history[-5:]  # Last 5 iterations
        }
        
        return stats
    
    def reset_labeled_indices(self, labeled_indices: Optional[List[int]] = None):
        """Reset or set labeled indices."""
        if labeled_indices is None:
            self.labeled_indices = set()
        else:
            self.labeled_indices = set(labeled_indices)
        
        logger.info(f"Reset labeled indices. Current count: {len(self.labeled_indices)}")


def main():
    """Demonstrate active learning capabilities."""
    logger.info("Demonstrating Active Learning capabilities...")
    
    logger.info("Active Learning Components:")
    logger.info("✓ Uncertainty-based sampling (entropy, margin, least confident)")
    logger.info("✓ Diversity-based sampling (k-means, core-set)")
    logger.info("✓ Combined selection strategies")
    logger.info("✓ Feature extraction and analysis")
    logger.info("✓ Selection history tracking")
    
    logger.info("\nActive Learning Benefits:")
    logger.info("• Reduce annotation costs by 50-80%")
    logger.info("• Focus on most informative samples")
    logger.info("• Balance uncertainty and diversity")
    logger.info("• Continuous model improvement")
    logger.info("• Adaptive to changing data distributions")
    
    logger.info("\nUse Cases:")
    logger.info("• New crop disease identification")
    logger.info("• Seasonal adaptation of models")
    logger.info("• Geographic domain adaptation")
    logger.info("• Rare event detection improvement")
    
    logger.info("\nActive learning system ready for deployment!")


if __name__ == "__main__":
    main()