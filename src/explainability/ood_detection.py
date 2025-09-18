"""
Out-of-Distribution Detection for Crop Health Models

Implements various OOD detection methods to identify novel
stress patterns and unseen conditions in agricultural data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutOfDistributionDetector:
    """Comprehensive OOD detection for agricultural models."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 methods: List[str] = ['mahalanobis', 'confidence', 'energy']):
        """
        Initialize OOD Detector.
        
        Args:
            model: Trained model for feature extraction
            device: Computation device
            methods: OOD detection methods to use
        """
        self.model = model.to(device).eval()
        self.device = device
        self.methods = methods
        
        # Storage for training statistics
        self.training_features = None
        self.feature_mean = None
        self.feature_cov = None
        self.feature_inv_cov = None
        
        # Method-specific parameters
        self.confidence_threshold = 0.9
        self.energy_threshold = None
        self.isolation_forest = None
        self.lof_detector = None
        
        # Feature extraction hook
        self.features = {}
        self.hooks = []
        self._register_feature_hooks()
    
    def _register_feature_hooks(self):
        """Register hooks to extract intermediate features."""
        def feature_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        # Register hooks for key layers
        target_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and 'classifier' not in name:
                target_layers.append(name)
        
        # Use the last few layers before classification
        if len(target_layers) >= 2:
            selected_layers = target_layers[-2:]
        else:
            selected_layers = target_layers
        
        for layer_name in selected_layers:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    handle = module.register_forward_hook(feature_hook(name))
                    self.hooks.append(handle)
                    logger.info(f"Registered feature hook for: {name}")
    
    def fit_training_distribution(self, 
                                 training_loader,
                                 max_samples: int = 10000) -> None:
        """
        Fit OOD detectors on training data distribution.
        
        Args:
            training_loader: Training data loader
            max_samples: Maximum samples to use for fitting
        """
        logger.info("Fitting OOD detectors on training distribution...")
        
        all_features = []
        all_predictions = []
        sample_count = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in training_loader:
                if sample_count >= max_samples:
                    break
                
                # Move data to device
                if isinstance(batch, dict):
                    if 'image_sequence' in batch:
                        inputs = (batch['image_sequence'].to(self.device),
                                batch['sensor_sequence'].to(self.device))
                    else:
                        inputs = batch['input'].to(self.device)
                else:
                    inputs = batch[0].to(self.device)
                
                # Forward pass
                try:
                    if isinstance(inputs, tuple):
                        outputs = self.model(*inputs)
                    else:
                        outputs = self.model(inputs)
                    
                    # Handle different output formats
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    elif isinstance(outputs, dict):
                        logits = outputs['predictions']
                    else:
                        logits = outputs
                    
                    # Extract features from hooks
                    batch_features = self._extract_features()
                    all_features.extend(batch_features)
                    
                    # Store predictions
                    probabilities = F.softmax(logits, dim=1)
                    all_predictions.extend(probabilities.cpu().numpy())
                    
                    sample_count += logits.size(0)
                    
                except Exception as e:
                    logger.warning(f"Failed to process batch: {e}")
                    continue
        
        # Convert to arrays
        self.training_features = np.array(all_features)
        training_predictions = np.array(all_predictions)
        
        logger.info(f"Collected {len(all_features)} training samples")
        
        # Fit different OOD detection methods
        if 'mahalanobis' in self.methods:
            self._fit_mahalanobis()
        
        if 'energy' in self.methods:
            self._fit_energy_threshold(training_predictions)
        
        if 'isolation_forest' in self.methods:
            self._fit_isolation_forest()
        
        if 'lof' in self.methods:
            self._fit_local_outlier_factor()
        
        logger.info("OOD detector fitting completed")
    
    def _extract_features(self) -> List[np.ndarray]:
        """Extract features from registered hooks."""
        features_list = []
        
        for layer_name, feature_tensor in self.features.items():
            # Flatten features
            batch_size = feature_tensor.size(0)
            flattened = feature_tensor.view(batch_size, -1)
            features_list.extend(flattened.cpu().numpy())
        
        return features_list
    
    def _fit_mahalanobis(self) -> None:
        """Fit Mahalanobis distance detector."""
        logger.info("Fitting Mahalanobis distance detector...")
        
        # Calculate mean and covariance
        self.feature_mean = np.mean(self.training_features, axis=0)
        
        # Use empirical covariance with regularization
        cov_estimator = EmpiricalCovariance(assume_centered=False)
        cov_estimator.fit(self.training_features)
        
        self.feature_cov = cov_estimator.covariance_
        
        # Compute inverse covariance with regularization
        try:
            self.feature_inv_cov = np.linalg.inv(self.feature_cov)
        except np.linalg.LinAlgError:
            # Add regularization if matrix is singular
            reg_param = 1e-6
            regularized_cov = self.feature_cov + reg_param * np.eye(self.feature_cov.shape[0])
            self.feature_inv_cov = np.linalg.inv(regularized_cov)
        
        logger.info("Mahalanobis detector fitted")
    
    def _fit_energy_threshold(self, training_predictions: np.ndarray) -> None:
        """Fit energy-based threshold."""
        logger.info("Fitting energy-based threshold...")
        
        # Calculate energy scores for training data
        energy_scores = []
        for pred in training_predictions:
            energy = -np.log(np.sum(np.exp(pred)))
            energy_scores.append(energy)
        
        # Set threshold at 95th percentile
        self.energy_threshold = np.percentile(energy_scores, 95)
        
        logger.info(f"Energy threshold set to: {self.energy_threshold:.4f}")
    
    def _fit_isolation_forest(self) -> None:
        """Fit Isolation Forest detector."""
        logger.info("Fitting Isolation Forest detector...")
        
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(self.training_features)
        
        logger.info("Isolation Forest detector fitted")
    
    def _fit_local_outlier_factor(self) -> None:
        """Fit Local Outlier Factor detector."""
        logger.info("Fitting Local Outlier Factor detector...")
        
        # LOF is fit during prediction, just store parameters
        self.lof_detector = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True
        )
        self.lof_detector.fit(self.training_features)
        
        logger.info("Local Outlier Factor detector fitted")
    
    def detect_ood(self, 
                   inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                   return_scores: bool = True) -> Dict[str, Any]:
        """
        Detect out-of-distribution samples.
        
        Args:
            inputs: Input data (tensor or tuple of tensors)
            return_scores: Whether to return detailed scores
            
        Returns:
            Dictionary with OOD detection results
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            if isinstance(inputs, tuple):
                inputs = tuple(inp.to(self.device) for inp in inputs)
                outputs = self.model(*inputs)
            else:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                logits = outputs['predictions']
            else:
                logits = outputs
            
            # Extract features
            sample_features = self._extract_features()
            sample_features = np.array(sample_features)
            
            # Get predictions
            probabilities = F.softmax(logits, dim=1)
            max_probs = torch.max(probabilities, dim=1)[0]
        
        # Apply different OOD detection methods
        results = {
            'is_ood': {},
            'scores': {} if return_scores else None,
            'predictions': probabilities.cpu().numpy(),
            'max_confidence': max_probs.cpu().numpy()
        }
        
        # Confidence-based detection
        if 'confidence' in self.methods:
            conf_ood = max_probs.cpu().numpy() < self.confidence_threshold
            results['is_ood']['confidence'] = conf_ood
            if return_scores:
                results['scores']['confidence'] = max_probs.cpu().numpy()
        
        # Mahalanobis distance
        if 'mahalanobis' in self.methods and self.feature_mean is not None:
            mahal_scores = self._compute_mahalanobis_scores(sample_features)
            mahal_threshold = np.percentile(mahal_scores, 95)  # Could be pre-computed
            mahal_ood = mahal_scores > mahal_threshold
            
            results['is_ood']['mahalanobis'] = mahal_ood
            if return_scores:
                results['scores']['mahalanobis'] = mahal_scores
        
        # Energy-based detection
        if 'energy' in self.methods and self.energy_threshold is not None:
            energy_scores = self._compute_energy_scores(probabilities.cpu().numpy())
            energy_ood = energy_scores > self.energy_threshold
            
            results['is_ood']['energy'] = energy_ood
            if return_scores:
                results['scores']['energy'] = energy_scores
        
        # Isolation Forest
        if 'isolation_forest' in self.methods and self.isolation_forest is not None:
            iso_predictions = self.isolation_forest.predict(sample_features)
            iso_ood = iso_predictions == -1
            
            results['is_ood']['isolation_forest'] = iso_ood
            if return_scores:
                iso_scores = self.isolation_forest.decision_function(sample_features)
                results['scores']['isolation_forest'] = iso_scores
        
        # Local Outlier Factor
        if 'lof' in self.methods and self.lof_detector is not None:
            lof_predictions = self.lof_detector.predict(sample_features)
            lof_ood = lof_predictions == -1
            
            results['is_ood']['lof'] = lof_ood
            if return_scores:
                lof_scores = self.lof_detector.decision_function(sample_features)
                results['scores']['lof'] = lof_scores
        
        # Ensemble decision (majority vote)
        if len(results['is_ood']) > 1:
            ood_votes = np.array(list(results['is_ood'].values())).astype(int)
            ensemble_ood = np.mean(ood_votes, axis=0) > 0.5
            results['is_ood']['ensemble'] = ensemble_ood
        
        return results
    
    def _compute_mahalanobis_scores(self, features: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances."""
        diff = features - self.feature_mean
        mahal_scores = np.sqrt(np.sum(diff @ self.feature_inv_cov * diff, axis=1))
        return mahal_scores
    
    def _compute_energy_scores(self, probabilities: np.ndarray) -> np.ndarray:
        """Compute energy scores."""
        energy_scores = []
        for prob in probabilities:
            energy = -np.log(np.sum(np.exp(prob)))
            energy_scores.append(energy)
        return np.array(energy_scores)
    
    def evaluate_ood_detection(self, 
                              in_distribution_loader,
                              out_distribution_loader) -> Dict[str, float]:
        """
        Evaluate OOD detection performance.
        
        Args:
            in_distribution_loader: In-distribution test data
            out_distribution_loader: Out-of-distribution test data
            
        Returns:
            Performance metrics for each method
        """
        logger.info("Evaluating OOD detection performance...")
        
        # Collect predictions for in-distribution data
        in_dist_results = []
        for batch in in_distribution_loader:
            try:
                if isinstance(batch, dict):
                    if 'image_sequence' in batch:
                        inputs = (batch['image_sequence'], batch['sensor_sequence'])
                    else:
                        inputs = batch['input']
                else:
                    inputs = batch[0]
                
                results = self.detect_ood(inputs, return_scores=True)
                in_dist_results.append(results)
            except Exception as e:
                logger.warning(f"Failed to process in-distribution batch: {e}")
        
        # Collect predictions for out-of-distribution data
        out_dist_results = []
        for batch in out_distribution_loader:
            try:
                if isinstance(batch, dict):
                    if 'image_sequence' in batch:
                        inputs = (batch['image_sequence'], batch['sensor_sequence'])
                    else:
                        inputs = batch['input']
                else:
                    inputs = batch[0]
                
                results = self.detect_ood(inputs, return_scores=True)
                out_dist_results.append(results)
            except Exception as e:
                logger.warning(f"Failed to process out-of-distribution batch: {e}")
        
        # Calculate metrics for each method
        metrics = {}
        
        for method in self.methods + ['ensemble']:
            if method == 'ensemble' and len(self.methods) <= 1:
                continue
            
            # Collect predictions
            in_dist_preds = []
            out_dist_preds = []
            
            for result in in_dist_results:
                if method in result['is_ood']:
                    in_dist_preds.extend(result['is_ood'][method])
            
            for result in out_dist_results:
                if method in result['is_ood']:
                    out_dist_preds.extend(result['is_ood'][method])
            
            if len(in_dist_preds) == 0 or len(out_dist_preds) == 0:
                continue
            
            # Calculate metrics
            in_dist_preds = np.array(in_dist_preds)
            out_dist_preds = np.array(out_dist_preds)
            
            # True negatives (in-dist correctly classified as in-dist)
            tn = np.sum(~in_dist_preds)
            # False positives (in-dist incorrectly classified as OOD)
            fp = np.sum(in_dist_preds)
            # True positives (OOD correctly classified as OOD)
            tp = np.sum(out_dist_preds)
            # False negatives (OOD incorrectly classified as in-dist)
            fn = np.sum(~out_dist_preds)
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[method] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1_score,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }
        
        logger.info("OOD detection evaluation completed")
        return metrics
    
    def cleanup(self):
        """Remove hooks and cleanup."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.features.clear()


class NoveltyDetector:
    """Specialized detector for novel crop stress patterns."""
    
    def __init__(self, 
                 base_detector: OutOfDistributionDetector,
                 spectral_importance_threshold: float = 0.1):
        """
        Initialize Novelty Detector.
        
        Args:
            base_detector: Base OOD detector
            spectral_importance_threshold: Threshold for spectral novelty
        """
        self.base_detector = base_detector
        self.spectral_threshold = spectral_importance_threshold
        
        # Storage for known stress patterns
        self.known_stress_patterns = {}
        self.spectral_signatures = {}
    
    def register_stress_pattern(self, 
                               pattern_name: str,
                               spectral_signature: np.ndarray,
                               description: str = "") -> None:
        """Register a known stress pattern."""
        self.known_stress_patterns[pattern_name] = {
            'signature': spectral_signature,
            'description': description
        }
        logger.info(f"Registered stress pattern: {pattern_name}")
    
    def detect_novel_stress(self, 
                           inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                           spectral_features: np.ndarray) -> Dict[str, Any]:
        """
        Detect novel stress patterns.
        
        Args:
            inputs: Input data
            spectral_features: Extracted spectral features
            
        Returns:
            Novel stress detection results
        """
        # First run standard OOD detection
        ood_results = self.base_detector.detect_ood(inputs)
        
        # Check for spectral novelty
        spectral_novelty = self._detect_spectral_novelty(spectral_features)
        
        # Combine results
        results = {
            'is_novel_stress': ood_results['is_ood'].get('ensemble', False) and spectral_novelty['is_novel'],
            'ood_results': ood_results,
            'spectral_novelty': spectral_novelty,
            'confidence': self._compute_novelty_confidence(ood_results, spectral_novelty)
        }
        
        return results
    
    def _detect_spectral_novelty(self, spectral_features: np.ndarray) -> Dict[str, Any]:
        """Detect spectral novelty compared to known patterns."""
        if len(self.known_stress_patterns) == 0:
            return {'is_novel': False, 'similarity_scores': {}}
        
        similarity_scores = {}
        min_distance = float('inf')
        
        for pattern_name, pattern_data in self.known_stress_patterns.items():
            signature = pattern_data['signature']
            
            # Compute spectral similarity (cosine similarity)
            similarity = np.dot(spectral_features, signature) / (
                np.linalg.norm(spectral_features) * np.linalg.norm(signature)
            )
            similarity_scores[pattern_name] = similarity
            min_distance = min(min_distance, 1 - similarity)
        
        is_novel = min_distance > self.spectral_threshold
        
        return {
            'is_novel': is_novel,
            'min_distance': min_distance,
            'similarity_scores': similarity_scores
        }
    
    def _compute_novelty_confidence(self, 
                                   ood_results: Dict[str, Any],
                                   spectral_novelty: Dict[str, Any]) -> float:
        """Compute confidence in novelty detection."""
        # Combine OOD confidence with spectral novelty confidence
        ood_confidence = 1.0 - np.mean(ood_results['max_confidence'])
        spectral_confidence = spectral_novelty.get('min_distance', 0.0)
        
        # Weighted combination
        combined_confidence = 0.6 * ood_confidence + 0.4 * spectral_confidence
        
        return combined_confidence


def main():
    """Demonstrate OOD detection capabilities."""
    logger.info("Demonstrating Out-of-Distribution Detection capabilities...")
    
    logger.info("OOD Detection Methods:")
    logger.info("✓ Mahalanobis distance in feature space")
    logger.info("✓ Confidence-based thresholding")
    logger.info("✓ Energy-based detection")
    logger.info("✓ Isolation Forest")
    logger.info("✓ Local Outlier Factor")
    logger.info("✓ Ensemble voting")
    
    logger.info("\nNovelty Detection Features:")
    logger.info("✓ Novel stress pattern identification")
    logger.info("✓ Spectral signature comparison")
    logger.info("✓ Known pattern registration")
    logger.info("✓ Confidence estimation")
    
    logger.info("\nUse Cases:")
    logger.info("• Detect new disease symptoms")
    logger.info("• Identify unknown stress conditions")
    logger.info("• Flag samples requiring expert review")
    logger.info("• Improve model robustness")
    
    logger.info("\nOOD detection module ready for deployment!")


if __name__ == "__main__":
    main()