"""
Out-of-Distribution (OOD) Detection Module
Detect when inputs are significantly different from training data
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class OODDetector:
    """Base class for out-of-distribution detection"""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.is_fitted = False
    
    def fit(self, X):
        """Fit the OOD detector on in-distribution data"""
        raise NotImplementedError
    
    def predict(self, X):
        """Predict if samples are OOD (1) or in-distribution (0)"""
        raise NotImplementedError
    
    def score(self, X):
        """Return OOD scores (higher = more likely OOD)"""
        raise NotImplementedError

class MahalanobisOOD(OODDetector):
    """Mahalanobis distance-based OOD detection"""
    
    def __init__(self, threshold=3.0):
        super().__init__(threshold)
        self.mean = None
        self.cov_inv = None
    
    def fit(self, X):
        """Fit Mahalanobis detector"""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        # Flatten if needed
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        self.mean = np.mean(X, axis=0)
        cov = np.cov(X.T)
        
        # Add regularization for numerical stability
        reg = 1e-6 * np.eye(cov.shape[0])
        cov_reg = cov + reg
        
        try:
            self.cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            self.cov_inv = np.linalg.pinv(cov_reg)
        
        self.is_fitted = True
        logger.info("Mahalanobis OOD detector fitted")
    
    def score(self, X):
        """Calculate Mahalanobis distances"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        diff = X - self.mean
        distances = np.sqrt(np.sum(diff @ self.cov_inv * diff, axis=1))
        return distances
    
    def predict(self, X):
        """Predict OOD samples"""
        scores = self.score(X)
        return (scores > self.threshold).astype(int)

class IsolationForestOOD(OODDetector):
    """Isolation Forest-based OOD detection"""
    
    def __init__(self, contamination=0.1, n_estimators=100):
        super().__init__()
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
    
    def fit(self, X):
        """Fit Isolation Forest"""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        self.model.fit(X)
        self.is_fitted = True
        logger.info("Isolation Forest OOD detector fitted")
    
    def score(self, X):
        """Get anomaly scores (lower = more anomalous)"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Return negative scores so higher = more OOD
        return -self.model.score_samples(X)
    
    def predict(self, X):
        """Predict OOD samples"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Convert to binary (1 = OOD, 0 = in-distribution)
        predictions = self.model.predict(X)
        return (predictions == -1).astype(int)

class OneClassSVMOOD(OODDetector):
    """One-Class SVM based OOD detection"""
    
    def __init__(self, nu=0.1, gamma='scale'):
        super().__init__()
        self.model = OneClassSVM(nu=nu, gamma=gamma)
        self.scaler = StandardScaler()
    
    def fit(self, X):
        """Fit One-Class SVM"""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        logger.info("One-Class SVM OOD detector fitted")
    
    def score(self, X):
        """Get decision scores"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.transform(X)
        return -self.model.decision_function(X_scaled)  # Negative for consistency
    
    def predict(self, X):
        """Predict OOD samples"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return (predictions == -1).astype(int)

class DeepOOD(OODDetector):
    """Deep learning-based OOD detection using feature representations"""
    
    def __init__(self, model, layer_name=None, method='mahalanobis'):
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self.method = method
        self.feature_extractor = None
        self.ood_detector = None
        self._setup_feature_extractor()
    
    def _setup_feature_extractor(self):
        """Setup feature extractor from specified layer"""
        if self.layer_name is None:
            # Use the layer before the final classifier
            modules = list(self.model.children())
            self.feature_extractor = nn.Sequential(*modules[:-1])
        else:
            # Extract features from specific layer
            def hook_fn(module, input, output):
                self.features = output
            
            for name, module in self.model.named_modules():
                if name == self.layer_name:
                    module.register_forward_hook(hook_fn)
                    break
    
    def _extract_features(self, X):
        """Extract features from the model"""
        self.model.eval()
        features = []
        
        with torch.no_grad():
            if self.layer_name is None:
                # Use feature extractor
                feat = self.feature_extractor(X)
                if feat.dim() > 2:
                    feat = feat.mean(dim=(2, 3))  # Global average pooling
                features = feat.cpu().numpy()
            else:
                # Use hook
                self.features = None
                _ = self.model(X)
                if self.features is not None:
                    feat = self.features
                    if feat.dim() > 2:
                        feat = feat.mean(dim=(2, 3))
                    features = feat.cpu().numpy()
        
        return features
    
    def fit(self, X):
        """Fit deep OOD detector"""
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        # Extract features
        features = self._extract_features(X)
        
        # Setup OOD detector based on method
        if self.method == 'mahalanobis':
            self.ood_detector = MahalanobisOOD()
        elif self.method == 'isolation_forest':
            self.ood_detector = IsolationForestOOD()
        elif self.method == 'one_class_svm':
            self.ood_detector = OneClassSVMOOD()
        
        self.ood_detector.fit(features)
        self.is_fitted = True
        logger.info(f"Deep OOD detector fitted using {self.method}")
    
    def score(self, X):
        """Score OOD likelihood"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        features = self._extract_features(X)
        return self.ood_detector.score(features)
    
    def predict(self, X):
        """Predict OOD samples"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        
        features = self._extract_features(X)
        return self.ood_detector.predict(features)

class EnsembleOOD(OODDetector):
    """Ensemble of multiple OOD detectors"""
    
    def __init__(self, detectors, voting='soft'):
        super().__init__()
        self.detectors = detectors
        self.voting = voting  # 'soft' or 'hard'
    
    def fit(self, X):
        """Fit all detectors"""
        for detector in self.detectors:
            detector.fit(X)
        self.is_fitted = True
        logger.info(f"Ensemble OOD detector fitted with {len(self.detectors)} detectors")
    
    def score(self, X):
        """Ensemble scoring"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        scores = []
        for detector in self.detectors:
            score = detector.score(X)
            # Normalize scores to [0, 1]
            score_norm = (score - score.min()) / (score.max() - score.min() + 1e-8)
            scores.append(score_norm)
        
        # Average scores
        ensemble_scores = np.mean(scores, axis=0)
        return ensemble_scores
    
    def predict(self, X):
        """Ensemble prediction"""
        if self.voting == 'soft':
            scores = self.score(X)
            return (scores > 0.5).astype(int)
        else:  # hard voting
            predictions = []
            for detector in self.detectors:
                pred = detector.predict(X)
                predictions.append(pred)
            
            # Majority vote
            ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
            return ensemble_pred

def create_ood_detector(detector_type='mahalanobis', **kwargs):
    """Factory function to create OOD detectors"""
    if detector_type == 'mahalanobis':
        return MahalanobisOOD(**kwargs)
    elif detector_type == 'isolation_forest':
        return IsolationForestOOD(**kwargs)
    elif detector_type == 'one_class_svm':
        return OneClassSVMOOD(**kwargs)
    elif detector_type == 'deep':
        return DeepOOD(**kwargs)
    elif detector_type == 'ensemble':
        return EnsembleOOD(**kwargs)
    else:
        raise ValueError(f"Unknown OOD detector type: {detector_type}")

def main():
    """Test OOD detection functionality"""
    logger.info("OOD detection module initialized")

if __name__ == "__main__":
    main()