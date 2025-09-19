"""
Active Learning Module
Intelligent sample selection for continuous model improvement
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, squareform
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ActiveLearningStrategy:
    """Base class for active learning strategies"""
    
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
    
    def select_samples(self, model, unlabeled_data, **kwargs):
        """Select samples for labeling"""
        raise NotImplementedError

class UncertaintyBasedSampling(ActiveLearningStrategy):
    """Uncertainty-based sample selection"""
    
    def __init__(self, batch_size=10, strategy='entropy'):
        super().__init__(batch_size)
        self.strategy = strategy
    
    def select_samples(self, model, unlabeled_data, **kwargs):
        """Select samples based on prediction uncertainty"""
        model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for i, data in enumerate(unlabeled_data):
                if isinstance(data, np.ndarray):
                    data = torch.FloatTensor(data).unsqueeze(0)
                
                output = model(data)
                probabilities = F.softmax(output, dim=1)
                
                if self.strategy == 'entropy':
                    # Entropy-based uncertainty
                    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
                    uncertainty = entropy.item()
                elif self.strategy == 'margin':
                    # Margin-based uncertainty
                    sorted_probs, _ = torch.sort(probabilities, descending=True)
                    margin = sorted_probs[0, 0] - sorted_probs[0, 1]
                    uncertainty = 1 - margin.item()
                elif self.strategy == 'least_confidence':
                    # Least confidence
                    max_prob = torch.max(probabilities, dim=1)[0]
                    uncertainty = 1 - max_prob.item()
                else:
                    raise ValueError(f"Unknown uncertainty strategy: {self.strategy}")
                
                uncertainties.append((i, uncertainty))
        
        # Sort by uncertainty (descending) and select top samples
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in uncertainties[:self.batch_size]]
        
        logger.info(f"Selected {len(selected_indices)} samples using {self.strategy} uncertainty")
        return selected_indices

class DiversityBasedSampling(ActiveLearningStrategy):
    """Diversity-based sample selection"""
    
    def __init__(self, batch_size=10, strategy='k_centers'):
        super().__init__(batch_size)
        self.strategy = strategy
    
    def select_samples(self, model, unlabeled_data, labeled_data=None, **kwargs):
        """Select diverse samples"""
        # Extract features
        features = self._extract_features(model, unlabeled_data)
        
        if self.strategy == 'k_centers':
            selected_indices = self._k_centers_selection(features)
        elif self.strategy == 'clustering':
            selected_indices = self._clustering_selection(features)
        else:
            raise ValueError(f"Unknown diversity strategy: {self.strategy}")
        
        logger.info(f"Selected {len(selected_indices)} samples using {self.strategy} diversity")
        return selected_indices
    
    def _extract_features(self, model, data):
        """Extract feature representations"""
        model.eval()
        features = []
        
        # Get features from the layer before classification
        def hook_fn(module, input, output):
            features.append(output.cpu().numpy())
        
        # Register hook on the last feature layer
        hook = None
        for name, module in model.named_modules():
            if 'classifier' in name or 'fc' in name:
                # Hook the previous layer
                break
            last_module = module
        
        if hasattr(model, 'features'):
            hook = model.features[-1].register_forward_hook(hook_fn)
        else:
            # Try to find appropriate layer
            modules = list(model.children())
            if len(modules) > 1:
                hook = modules[-2].register_forward_hook(hook_fn)
        
        with torch.no_grad():
            for data_item in data:
                if isinstance(data_item, np.ndarray):
                    data_item = torch.FloatTensor(data_item).unsqueeze(0)
                _ = model(data_item)
        
        if hook:
            hook.remove()
        
        if features:
            return np.vstack(features)
        else:
            # Fallback: use raw data
            return np.array([d.flatten() if isinstance(d, np.ndarray) else d.cpu().numpy().flatten() 
                           for d in data])
    
    def _k_centers_selection(self, features):
        """K-centers greedy selection"""
        n_samples = len(features)
        selected = []
        remaining = list(range(n_samples))
        
        # Start with random sample
        first_idx = np.random.choice(remaining)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Greedily select samples that are farthest from selected set
        for _ in range(min(self.batch_size - 1, len(remaining))):
            max_min_distance = -1
            best_idx = None
            
            for idx in remaining:
                # Find minimum distance to selected samples
                min_distance = min([
                    np.linalg.norm(features[idx] - features[sel_idx]) 
                    for sel_idx in selected
                ])
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return selected
    
    def _clustering_selection(self, features):
        """Clustering-based selection"""
        n_clusters = min(self.batch_size, len(features))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)
        
        # Select one sample from each cluster (closest to centroid)
        selected = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            if len(cluster_indices) > 0:
                # Find sample closest to centroid
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = [np.linalg.norm(features[idx] - centroid) for idx in cluster_indices]
                closest_idx = cluster_indices[np.argmin(distances)]
                selected.append(closest_idx)
        
        return selected

class QueryByCommittee(ActiveLearningStrategy):
    """Query by committee active learning"""
    
    def __init__(self, batch_size=10, committee_models=None):
        super().__init__(batch_size)
        self.committee_models = committee_models or []
    
    def select_samples(self, model, unlabeled_data, **kwargs):
        """Select samples based on committee disagreement"""
        if not self.committee_models:
            logger.warning("No committee models provided, falling back to uncertainty sampling")
            uncertainty_strategy = UncertaintyBasedSampling(self.batch_size)
            return uncertainty_strategy.select_samples(model, unlabeled_data)
        
        disagreements = []
        
        for i, data in enumerate(unlabeled_data):
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data).unsqueeze(0)
            
            # Get predictions from all committee members
            predictions = []
            for committee_model in self.committee_models:
                committee_model.eval()
                with torch.no_grad():
                    output = committee_model(data)
                    prob = F.softmax(output, dim=1)
                    predictions.append(prob.cpu().numpy())
            
            # Calculate disagreement (variance in predictions)
            predictions = np.array(predictions)
            variance = np.var(predictions, axis=0).mean()
            disagreements.append((i, variance))
        
        # Sort by disagreement (descending) and select top samples
        disagreements.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in disagreements[:self.batch_size]]
        
        logger.info(f"Selected {len(selected_indices)} samples using committee disagreement")
        return selected_indices

class HybridActiveLearning(ActiveLearningStrategy):
    """Hybrid approach combining multiple strategies"""
    
    def __init__(self, batch_size=10, uncertainty_weight=0.5, diversity_weight=0.5):
        super().__init__(batch_size)
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        
        self.uncertainty_strategy = UncertaintyBasedSampling(batch_size * 2)  # Get more candidates
        self.diversity_strategy = DiversityBasedSampling(batch_size)
    
    def select_samples(self, model, unlabeled_data, **kwargs):
        """Hybrid selection combining uncertainty and diversity"""
        
        # Get uncertainty-based candidates
        uncertain_indices = self.uncertainty_strategy.select_samples(model, unlabeled_data)
        
        # Filter unlabeled data to uncertain candidates
        uncertain_data = [unlabeled_data[i] for i in uncertain_indices]
        
        # Apply diversity selection on uncertain candidates
        diverse_indices = self.diversity_strategy.select_samples(model, uncertain_data)
        
        # Map back to original indices
        selected_indices = [uncertain_indices[i] for i in diverse_indices]
        
        logger.info(f"Selected {len(selected_indices)} samples using hybrid approach")
        return selected_indices

class ActiveLearningManager:
    """Manages active learning process"""
    
    def __init__(self, model, strategy='uncertainty', batch_size=10):
        self.model = model
        self.batch_size = batch_size
        
        # Initialize strategy
        if strategy == 'uncertainty':
            self.strategy = UncertaintyBasedSampling(batch_size)
        elif strategy == 'diversity':
            self.strategy = DiversityBasedSampling(batch_size)
        elif strategy == 'committee':
            self.strategy = QueryByCommittee(batch_size)
        elif strategy == 'hybrid':
            self.strategy = HybridActiveLearning(batch_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.labeled_data = []
        self.unlabeled_data = []
        self.selection_history = []
    
    def add_unlabeled_data(self, data):
        """Add unlabeled data to pool"""
        if isinstance(data, list):
            self.unlabeled_data.extend(data)
        else:
            self.unlabeled_data.append(data)
        
        logger.info(f"Added {len(data) if isinstance(data, list) else 1} unlabeled samples")
    
    def select_for_labeling(self):
        """Select samples for labeling"""
        if len(self.unlabeled_data) == 0:
            logger.warning("No unlabeled data available")
            return []
        
        selected_indices = self.strategy.select_samples(self.model, self.unlabeled_data)
        
        # Store selection history
        selection_record = {
            'timestamp': datetime.now().isoformat(),
            'strategy': type(self.strategy).__name__,
            'selected_count': len(selected_indices),
            'total_unlabeled': len(self.unlabeled_data)
        }
        self.selection_history.append(selection_record)
        
        return selected_indices
    
    def add_labeled_samples(self, indices, labels):
        """Add newly labeled samples to training set"""
        for idx, label in zip(indices, labels):
            if idx < len(self.unlabeled_data):
                sample = self.unlabeled_data[idx]
                self.labeled_data.append((sample, label))
        
        # Remove from unlabeled pool (in reverse order to maintain indices)
        for idx in sorted(indices, reverse=True):
            if idx < len(self.unlabeled_data):
                del self.unlabeled_data[idx]
        
        logger.info(f"Added {len(indices)} labeled samples")
    
    def get_statistics(self):
        """Get active learning statistics"""
        return {
            'labeled_samples': len(self.labeled_data),
            'unlabeled_samples': len(self.unlabeled_data),
            'selection_rounds': len(self.selection_history),
            'strategy': type(self.strategy).__name__,
            'batch_size': self.batch_size
        }

def create_active_learning_strategy(strategy_type='uncertainty', **kwargs):
    """Factory function to create active learning strategies"""
    if strategy_type == 'uncertainty':
        return UncertaintyBasedSampling(**kwargs)
    elif strategy_type == 'diversity':
        return DiversityBasedSampling(**kwargs)
    elif strategy_type == 'committee':
        return QueryByCommittee(**kwargs)
    elif strategy_type == 'hybrid':
        return HybridActiveLearning(**kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

def main():
    """Test active learning functionality"""
    logger.info("Active learning module initialized")

if __name__ == "__main__":
    main()