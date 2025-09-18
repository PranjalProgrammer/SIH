"""
Model Drift Detection and Performance Monitoring

Detects changes in data distribution and model performance
to trigger retraining when necessary.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects various types of data and concept drift."""
    
    def __init__(self, 
                 reference_window_size: int = 1000,
                 detection_window_size: int = 100,
                 significance_level: float = 0.05):
        """
        Initialize Drift Detector.
        
        Args:
            reference_window_size: Size of reference window for comparison
            detection_window_size: Size of detection window
            significance_level: Statistical significance level for tests
        """
        self.reference_window_size = reference_window_size
        self.detection_window_size = detection_window_size
        self.significance_level = significance_level
        
        # Storage for reference data
        self.reference_features = None
        self.reference_predictions = None
        self.reference_labels = None
        
        # Drift detection history
        self.drift_history = []
        
    def set_reference_data(self, 
                          features: np.ndarray,
                          predictions: Optional[np.ndarray] = None,
                          labels: Optional[np.ndarray] = None):
        """
        Set reference data for drift detection.
        
        Args:
            features: Reference feature data
            predictions: Reference model predictions
            labels: Reference true labels
        """
        # Sample if data is too large
        if len(features) > self.reference_window_size:
            indices = np.random.choice(len(features), self.reference_window_size, replace=False)
            self.reference_features = features[indices]
            if predictions is not None:
                self.reference_predictions = predictions[indices]
            if labels is not None:
                self.reference_labels = labels[indices]
        else:
            self.reference_features = features
            self.reference_predictions = predictions
            self.reference_labels = labels
        
        logger.info(f"Set reference data with {len(self.reference_features)} samples")
    
    def detect_feature_drift(self, 
                           current_features: np.ndarray,
                           method: str = 'ks_test') -> Dict[str, Any]:
        """
        Detect drift in input features.
        
        Args:
            current_features: Current feature data
            method: Drift detection method ('ks_test', 'psi', 'mmd')
            
        Returns:
            Drift detection results
        """
        if self.reference_features is None:
            raise ValueError("Reference data not set. Call set_reference_data first.")
        
        logger.info(f"Detecting feature drift using {method}")
        
        if method == 'ks_test':
            return self._ks_test_drift(current_features)
        elif method == 'psi':
            return self._psi_drift(current_features)
        elif method == 'mmd':
            return self._mmd_drift(current_features)
        else:
            raise ValueError(f"Unknown drift detection method: {method}")
    
    def _ks_test_drift(self, current_features: np.ndarray) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test for feature drift."""
        n_features = min(self.reference_features.shape[1], current_features.shape[1])
        
        drift_results = {
            'method': 'ks_test',
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False,
            'feature_drifts': [],
            'overall_p_value': 1.0,
            'significant_features': []
        }
        
        p_values = []
        
        for i in range(n_features):
            ref_feature = self.reference_features[:, i]
            cur_feature = current_features[:, i]
            
            # Perform KS test
            statistic, p_value = stats.ks_2samp(ref_feature, cur_feature)
            
            feature_drift = {
                'feature_index': i,
                'ks_statistic': float(statistic),
                'p_value': float(p_value),
                'drift_detected': p_value < self.significance_level
            }
            
            drift_results['feature_drifts'].append(feature_drift)
            p_values.append(p_value)
            
            if p_value < self.significance_level:
                drift_results['significant_features'].append(i)
        
        # Overall drift assessment
        # Use Bonferroni correction for multiple testing
        adjusted_alpha = self.significance_level / n_features
        drift_results['drift_detected'] = any(p < adjusted_alpha for p in p_values)
        drift_results['overall_p_value'] = min(p_values) if p_values else 1.0
        
        return drift_results
    
    def _psi_drift(self, current_features: np.ndarray) -> Dict[str, Any]:
        """Population Stability Index (PSI) for feature drift."""
        n_features = min(self.reference_features.shape[1], current_features.shape[1])
        
        drift_results = {
            'method': 'psi',
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False,
            'feature_drifts': [],
            'overall_psi': 0.0,
            'significant_features': []
        }
        
        psi_threshold = 0.2  # Standard PSI threshold
        psi_values = []
        
        for i in range(n_features):
            ref_feature = self.reference_features[:, i]
            cur_feature = current_features[:, i]
            
            # Calculate PSI
            psi_value = self._calculate_psi(ref_feature, cur_feature)
            
            feature_drift = {
                'feature_index': i,
                'psi_value': float(psi_value),
                'drift_detected': psi_value > psi_threshold
            }
            
            drift_results['feature_drifts'].append(feature_drift)
            psi_values.append(psi_value)
            
            if psi_value > psi_threshold:
                drift_results['significant_features'].append(i)
        
        # Overall drift assessment
        drift_results['overall_psi'] = np.mean(psi_values)
        drift_results['drift_detected'] = drift_results['overall_psi'] > psi_threshold
        
        return drift_results
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        # Create bins based on reference data
        bin_edges = np.histogram_bin_edges(reference, bins=bins)
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to proportions
        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)
        
        # Avoid division by zero
        ref_props = np.where(ref_props == 0, 1e-6, ref_props)
        cur_props = np.where(cur_props == 0, 1e-6, cur_props)
        
        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        return psi
    
    def _mmd_drift(self, current_features: np.ndarray) -> Dict[str, Any]:
        """Maximum Mean Discrepancy (MMD) for feature drift."""
        # Simplified MMD implementation
        def rbf_kernel(X, Y, gamma=1.0):
            """RBF kernel for MMD calculation."""
            X_norm = np.sum(X**2, axis=1, keepdims=True)
            Y_norm = np.sum(Y**2, axis=1, keepdims=True)
            
            distances = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
            return np.exp(-gamma * distances)
        
        # Sample data for computational efficiency
        n_samples = min(500, len(self.reference_features), len(current_features))
        
        ref_sample = self.reference_features[np.random.choice(len(self.reference_features), n_samples)]
        cur_sample = current_features[np.random.choice(len(current_features), n_samples)]
        
        # Calculate MMD
        K_XX = rbf_kernel(ref_sample, ref_sample)
        K_YY = rbf_kernel(cur_sample, cur_sample)
        K_XY = rbf_kernel(ref_sample, cur_sample)
        
        mmd_squared = (np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY))
        mmd_value = np.sqrt(max(mmd_squared, 0))
        
        # Threshold for drift detection (empirically determined)
        mmd_threshold = 0.1
        
        drift_results = {
            'method': 'mmd',
            'timestamp': datetime.now().isoformat(),
            'mmd_value': float(mmd_value),
            'drift_detected': mmd_value > mmd_threshold,
            'threshold': mmd_threshold
        }
        
        return drift_results
    
    def detect_prediction_drift(self, 
                              current_predictions: np.ndarray,
                              current_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect drift in model predictions.
        
        Args:
            current_predictions: Current model predictions
            current_labels: Current true labels (if available)
            
        Returns:
            Prediction drift results
        """
        if self.reference_predictions is None:
            raise ValueError("Reference predictions not set")
        
        logger.info("Detecting prediction drift")
        
        # Compare prediction distributions
        drift_results = {
            'method': 'prediction_drift',
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False
        }
        
        # KS test on prediction probabilities
        if current_predictions.ndim > 1:  # Probability distributions
            for class_idx in range(current_predictions.shape[1]):
                ref_probs = self.reference_predictions[:, class_idx]
                cur_probs = current_predictions[:, class_idx]
                
                statistic, p_value = stats.ks_2samp(ref_probs, cur_probs)
                
                if p_value < self.significance_level:
                    drift_results['drift_detected'] = True
                    drift_results[f'class_{class_idx}_p_value'] = float(p_value)
        
        # Compare accuracy if labels available
        if current_labels is not None and self.reference_labels is not None:
            ref_accuracy = accuracy_score(self.reference_labels, 
                                        np.argmax(self.reference_predictions, axis=1))
            cur_accuracy = accuracy_score(current_labels,
                                        np.argmax(current_predictions, axis=1))
            
            accuracy_drop = ref_accuracy - cur_accuracy
            drift_results['accuracy_drop'] = float(accuracy_drop)
            drift_results['reference_accuracy'] = float(ref_accuracy)
            drift_results['current_accuracy'] = float(cur_accuracy)
            
            # Flag significant accuracy drop
            if accuracy_drop > 0.05:  # 5% threshold
                drift_results['significant_performance_drop'] = True
        
        return drift_results
    
    def update_drift_history(self, drift_result: Dict[str, Any]):
        """Update drift detection history."""
        self.drift_history.append(drift_result)
        
        # Keep only last 100 records
        if len(self.drift_history) > 100:
            self.drift_history = self.drift_history[-100:]
    
    def get_drift_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of drift detections over specified period."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        recent_drifts = [
            drift for drift in self.drift_history
            if datetime.fromisoformat(drift['timestamp']) > cutoff_time
        ]
        
        summary = {
            'period_days': days,
            'total_detections': len(recent_drifts),
            'drift_detected_count': sum(1 for d in recent_drifts if d.get('drift_detected', False)),
            'drift_rate': 0.0,
            'methods_used': list(set(d.get('method', 'unknown') for d in recent_drifts))
        }
        
        if summary['total_detections'] > 0:
            summary['drift_rate'] = summary['drift_detected_count'] / summary['total_detections']
        
        return summary


class PerformanceMonitor:
    """Monitor model performance over time."""
    
    def __init__(self, 
                 window_size: int = 100,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize Performance Monitor.
        
        Args:
            window_size: Size of sliding window for metrics
            alert_thresholds: Thresholds for performance alerts
        """
        self.window_size = window_size
        self.alert_thresholds = alert_thresholds or {
            'accuracy_drop': 0.05,
            'f1_drop': 0.05,
            'precision_drop': 0.05,
            'recall_drop': 0.05
        }
        
        # Performance history
        self.performance_history = []
        self.baseline_metrics = None
        
    def set_baseline_performance(self, 
                               predictions: np.ndarray,
                               labels: np.ndarray):
        """Set baseline performance metrics."""
        self.baseline_metrics = self._calculate_metrics(predictions, labels)
        logger.info(f"Set baseline performance: {self.baseline_metrics}")
    
    def _calculate_metrics(self, 
                          predictions: np.ndarray,
                          labels: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Handle probability predictions
        if predictions.ndim > 1:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = predictions
        
        metrics = {
            'accuracy': accuracy_score(labels, pred_classes),
            'precision': precision_score(labels, pred_classes, average='weighted', zero_division=0),
            'recall': recall_score(labels, pred_classes, average='weighted', zero_division=0),
            'f1_score': f1_score(labels, pred_classes, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def update_performance(self, 
                          predictions: np.ndarray,
                          labels: np.ndarray,
                          timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Update performance metrics and check for alerts.
        
        Args:
            predictions: Model predictions
            labels: True labels
            timestamp: Timestamp for the metrics
            
        Returns:
            Performance update results including alerts
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate current metrics
        current_metrics = self._calculate_metrics(predictions, labels)
        
        # Add to history
        performance_record = {
            'timestamp': timestamp.isoformat(),
            'metrics': current_metrics,
            'sample_count': len(predictions)
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > self.window_size * 2:
            self.performance_history = self.performance_history[-self.window_size * 2:]
        
        # Check for performance alerts
        alerts = self._check_performance_alerts(current_metrics)
        
        result = {
            'current_metrics': current_metrics,
            'alerts': alerts,
            'timestamp': timestamp.isoformat()
        }
        
        if alerts:
            logger.warning(f"Performance alerts triggered: {alerts}")
        
        return result
    
    def _check_performance_alerts(self, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for performance degradation alerts."""
        alerts = []
        
        if self.baseline_metrics is None:
            return alerts
        
        for metric_name, current_value in current_metrics.items():
            baseline_value = self.baseline_metrics.get(metric_name, 0)
            drop = baseline_value - current_value
            
            threshold_key = f"{metric_name}_drop"
            if threshold_key in self.alert_thresholds:
                threshold = self.alert_thresholds[threshold_key]
                
                if drop > threshold:
                    alert = {
                        'type': 'performance_degradation',
                        'metric': metric_name,
                        'baseline_value': baseline_value,
                        'current_value': current_value,
                        'drop': drop,
                        'threshold': threshold,
                        'severity': 'high' if drop > threshold * 2 else 'medium'
                    }
                    alerts.append(alert)
        
        return alerts
    
    def get_performance_trend(self, days: int = 7) -> Dict[str, Any]:
        """Get performance trend over specified period."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        recent_records = [
            record for record in self.performance_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_time
        ]
        
        if not recent_records:
            return {'error': 'No recent performance data available'}
        
        # Calculate trends
        metrics_over_time = {}
        for metric_name in recent_records[0]['metrics'].keys():
            values = [record['metrics'][metric_name] for record in recent_records]
            
            # Calculate trend (simple linear regression slope)
            x = np.arange(len(values))
            if len(values) > 1:
                slope, _, _, _, _ = stats.linregress(x, values)
                trend = 'improving' if slope > 0.001 else 'declining' if slope < -0.001 else 'stable'
            else:
                slope = 0
                trend = 'stable'
            
            metrics_over_time[metric_name] = {
                'current_value': values[-1],
                'trend': trend,
                'slope': slope,
                'min_value': min(values),
                'max_value': max(values),
                'mean_value': np.mean(values)
            }
        
        return {
            'period_days': days,
            'records_count': len(recent_records),
            'metrics_trend': metrics_over_time
        }


def main():
    """Demonstrate drift detection capabilities."""
    logger.info("Demonstrating Drift Detection and Performance Monitoring...")
    
    logger.info("Drift Detection Methods:")
    logger.info("✓ Kolmogorov-Smirnov test for feature drift")
    logger.info("✓ Population Stability Index (PSI)")
    logger.info("✓ Maximum Mean Discrepancy (MMD)")
    logger.info("✓ Prediction distribution drift")
    logger.info("✓ Performance-based drift detection")
    
    logger.info("\nPerformance Monitoring Features:")
    logger.info("✓ Real-time performance tracking")
    logger.info("✓ Baseline comparison")
    logger.info("✓ Automated alert system")
    logger.info("✓ Performance trend analysis")
    logger.info("✓ Configurable thresholds")
    
    logger.info("\nBenefits:")
    logger.info("• Early detection of model degradation")
    logger.info("• Proactive retraining triggers")
    logger.info("• Maintain model accuracy over time")
    logger.info("• Reduce manual monitoring effort")
    logger.info("• Ensure reliable predictions")
    
    logger.info("\nDrift detection and monitoring system ready!")


if __name__ == "__main__":
    main()