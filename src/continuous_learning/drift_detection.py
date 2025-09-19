"""
Concept Drift Detection Module
Detect when data distribution changes over time
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from collections import deque
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DriftDetector:
    """Base class for drift detection"""
    
    def __init__(self, window_size=100, threshold=0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_data = None
        self.is_fitted = False
    
    def fit(self, reference_data):
        """Fit detector on reference data"""
        raise NotImplementedError
    
    def detect(self, new_data):
        """Detect drift in new data"""
        raise NotImplementedError
    
    def update(self, new_data):
        """Update detector with new data"""
        pass

class StatisticalDriftDetector(DriftDetector):
    """Statistical test-based drift detection"""
    
    def __init__(self, window_size=100, threshold=0.05, test='ks'):
        super().__init__(window_size, threshold)
        self.test = test
        self.reference_stats = {}
    
    def fit(self, reference_data):
        """Fit on reference data"""
        if isinstance(reference_data, torch.Tensor):
            reference_data = reference_data.cpu().numpy()
        
        # Flatten if multi-dimensional
        if reference_data.ndim > 2:
            reference_data = reference_data.reshape(reference_data.shape[0], -1)
        
        self.reference_data = reference_data
        
        # Compute reference statistics
        self.reference_stats = {
            'mean': np.mean(reference_data, axis=0),
            'std': np.std(reference_data, axis=0),
            'distribution': reference_data
        }
        
        self.is_fitted = True
        logger.info("Statistical drift detector fitted")
    
    def detect(self, new_data):
        """Detect drift using statistical tests"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if isinstance(new_data, torch.Tensor):
            new_data = new_data.cpu().numpy()
        
        if new_data.ndim > 2:
            new_data = new_data.reshape(new_data.shape[0], -1)
        
        drift_detected = False
        p_values = []
        
        # Test each feature
        for i in range(min(new_data.shape[1], self.reference_data.shape[1])):
            ref_feature = self.reference_data[:, i]
            new_feature = new_data[:, i]
            
            if self.test == 'ks':
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(ref_feature, new_feature)
            elif self.test == 'mann_whitney':
                # Mann-Whitney U test
                statistic, p_value = stats.mannwhitneyu(ref_feature, new_feature)
            elif self.test == 't_test':
                # T-test
                statistic, p_value = stats.ttest_ind(ref_feature, new_feature)
            else:
                raise ValueError(f"Unknown test: {self.test}")
            
            p_values.append(p_value)
            
            if p_value < self.threshold:
                drift_detected = True
        
        result = {
            'drift_detected': drift_detected,
            'p_values': p_values,
            'min_p_value': min(p_values),
            'test_used': self.test,
            'threshold': self.threshold
        }
        
        return result

class PerformanceDriftDetector(DriftDetector):
    """Performance-based drift detection"""
    
    def __init__(self, window_size=100, threshold=0.1):
        super().__init__(window_size, threshold)
        self.performance_history = deque(maxlen=window_size)
        self.baseline_performance = None
    
    def fit(self, model, reference_data, reference_labels):
        """Fit detector using model performance on reference data"""
        if isinstance(reference_data, torch.Tensor):
            reference_data = reference_data.cpu().numpy()
        if isinstance(reference_labels, torch.Tensor):
            reference_labels = reference_labels.cpu().numpy()
        
        # Calculate baseline performance
        model.eval()
        with torch.no_grad():
            if isinstance(reference_data, np.ndarray):
                reference_data = torch.FloatTensor(reference_data)
            
            outputs = model(reference_data)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        self.baseline_performance = accuracy_score(reference_labels, predictions)
        self.is_fitted = True
        
        logger.info(f"Performance drift detector fitted with baseline accuracy: {self.baseline_performance:.4f}")
    
    def detect(self, model, new_data, new_labels):
        """Detect drift based on performance degradation"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if isinstance(new_data, np.ndarray):
            new_data = torch.FloatTensor(new_data)
        if isinstance(new_labels, torch.Tensor):
            new_labels = new_labels.cpu().numpy()
        
        # Calculate current performance
        model.eval()
        with torch.no_grad():
            outputs = model(new_data)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        current_performance = accuracy_score(new_labels, predictions)
        self.performance_history.append(current_performance)
        
        # Check for drift
        performance_drop = self.baseline_performance - current_performance
        drift_detected = performance_drop > self.threshold
        
        # Calculate trend if we have enough history
        trend = None
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(list(self.performance_history)[-10:])
            older_performance = np.mean(list(self.performance_history)[:-10])
            trend = recent_performance - older_performance
        
        result = {
            'drift_detected': drift_detected,
            'current_performance': current_performance,
            'baseline_performance': self.baseline_performance,
            'performance_drop': performance_drop,
            'threshold': self.threshold,
            'trend': trend
        }
        
        return result
    
    def update(self, new_performance):
        """Update performance history"""
        self.performance_history.append(new_performance)

class FeatureDriftDetector(DriftDetector):
    """Feature-based drift detection using PCA"""
    
    def __init__(self, window_size=100, threshold=0.1, n_components=10):
        super().__init__(window_size, threshold)
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.reference_features = None
    
    def fit(self, reference_data):
        """Fit PCA on reference data"""
        if isinstance(reference_data, torch.Tensor):
            reference_data = reference_data.cpu().numpy()
        
        if reference_data.ndim > 2:
            reference_data = reference_data.reshape(reference_data.shape[0], -1)
        
        # Standardize and fit PCA
        reference_scaled = self.scaler.fit_transform(reference_data)
        self.reference_features = self.pca.fit_transform(reference_scaled)
        
        self.is_fitted = True
        logger.info(f"Feature drift detector fitted with {self.n_components} components")
    
    def detect(self, new_data):
        """Detect drift in feature space"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if isinstance(new_data, torch.Tensor):
            new_data = new_data.cpu().numpy()
        
        if new_data.ndim > 2:
            new_data = new_data.reshape(new_data.shape[0], -1)
        
        # Transform new data
        new_scaled = self.scaler.transform(new_data)
        new_features = self.pca.transform(new_scaled)
        
        # Compare distributions in PCA space
        drift_scores = []
        for i in range(self.n_components):
            ref_component = self.reference_features[:, i]
            new_component = new_features[:, i]
            
            # Use KL divergence or similar measure
            # For simplicity, using mean difference normalized by std
            ref_mean, ref_std = np.mean(ref_component), np.std(ref_component)
            new_mean = np.mean(new_component)
            
            drift_score = abs(new_mean - ref_mean) / (ref_std + 1e-8)
            drift_scores.append(drift_score)
        
        max_drift_score = max(drift_scores)
        drift_detected = max_drift_score > self.threshold
        
        result = {
            'drift_detected': drift_detected,
            'drift_scores': drift_scores,
            'max_drift_score': max_drift_score,
            'threshold': self.threshold,
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist()
        }
        
        return result

class EnsembleDriftDetector(DriftDetector):
    """Ensemble of multiple drift detectors"""
    
    def __init__(self, detectors, voting='majority'):
        self.detectors = detectors
        self.voting = voting  # 'majority', 'unanimous', 'any'
    
    def fit(self, *args, **kwargs):
        """Fit all detectors"""
        for detector in self.detectors:
            detector.fit(*args, **kwargs)
        
        self.is_fitted = True
        logger.info(f"Ensemble drift detector fitted with {len(self.detectors)} detectors")
    
    def detect(self, *args, **kwargs):
        """Ensemble drift detection"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        results = []
        drift_votes = []
        
        for detector in self.detectors:
            result = detector.detect(*args, **kwargs)
            results.append(result)
            drift_votes.append(result['drift_detected'])
        
        # Combine votes
        if self.voting == 'majority':
            drift_detected = sum(drift_votes) > len(drift_votes) / 2
        elif self.voting == 'unanimous':
            drift_detected = all(drift_votes)
        elif self.voting == 'any':
            drift_detected = any(drift_votes)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting}")
        
        ensemble_result = {
            'drift_detected': drift_detected,
            'individual_results': results,
            'votes': drift_votes,
            'voting_strategy': self.voting,
            'vote_count': sum(drift_votes)
        }
        
        return ensemble_result

class DriftMonitor:
    """Continuous drift monitoring system"""
    
    def __init__(self, detector, model=None, alert_threshold=3):
        self.detector = detector
        self.model = model
        self.alert_threshold = alert_threshold
        
        self.drift_history = []
        self.alert_count = 0
        self.last_alert_time = None
    
    def monitor_batch(self, new_data, labels=None):
        """Monitor a batch of new data"""
        try:
            if hasattr(self.detector, 'detect'):
                if labels is not None and isinstance(self.detector, PerformanceDriftDetector):
                    result = self.detector.detect(self.model, new_data, labels)
                else:
                    result = self.detector.detect(new_data)
            else:
                raise ValueError("Detector does not have detect method")
            
            # Add timestamp
            result['timestamp'] = datetime.now().isoformat()
            result['batch_size'] = len(new_data)
            
            # Store result
            self.drift_history.append(result)
            
            # Check for alerts
            if result['drift_detected']:
                self.alert_count += 1
                self.last_alert_time = datetime.now()
                
                if self.alert_count >= self.alert_threshold:
                    self._trigger_alert(result)
            else:
                # Reset alert count on non-drift detection
                self.alert_count = max(0, self.alert_count - 1)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in drift monitoring: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _trigger_alert(self, drift_result):
        """Trigger drift alert"""
        alert = {
            'type': 'concept_drift',
            'severity': 'high' if self.alert_count >= self.alert_threshold * 2 else 'medium',
            'message': f"Concept drift detected {self.alert_count} times",
            'drift_result': drift_result,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.warning(f"DRIFT ALERT: {alert['message']}")
        
        # Here you would typically send notifications, trigger retraining, etc.
        return alert
    
    def get_drift_summary(self, days=7):
        """Get drift summary for recent period"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        recent_history = [
            h for h in self.drift_history 
            if datetime.fromisoformat(h['timestamp']) > cutoff_time
        ]
        
        if not recent_history:
            return {'message': 'No recent drift history'}
        
        drift_count = sum(1 for h in recent_history if h.get('drift_detected', False))
        
        summary = {
            'total_batches': len(recent_history),
            'drift_detections': drift_count,
            'drift_rate': drift_count / len(recent_history),
            'alert_count': self.alert_count,
            'last_alert': self.last_alert_time.isoformat() if self.last_alert_time else None,
            'period_days': days
        }
        
        return summary

def create_drift_detector(detector_type='statistical', **kwargs):
    """Factory function to create drift detectors"""
    if detector_type == 'statistical':
        return StatisticalDriftDetector(**kwargs)
    elif detector_type == 'performance':
        return PerformanceDriftDetector(**kwargs)
    elif detector_type == 'feature':
        return FeatureDriftDetector(**kwargs)
    elif detector_type == 'ensemble':
        return EnsembleDriftDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")

def main():
    """Test drift detection functionality"""
    logger.info("Drift detection module initialized")

if __name__ == "__main__":
    main()