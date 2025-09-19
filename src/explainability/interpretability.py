"""
Model Interpretability Module
SHAP, LIME, and other interpretability methods for agricultural AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
try:
    import lime
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

import torch
import logging

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """SHAP-based model explanations"""
    
    def __init__(self, model, background_data=None):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        self.model = model
        self.background_data = background_data
        self.explainer = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """Setup SHAP explainer based on model type"""
        if hasattr(self.model, 'predict_proba'):
            # Sklearn-like model
            self.explainer = shap.Explainer(self.model, self.background_data)
        else:
            # Deep learning model
            def model_predict(x):
                with torch.no_grad():
                    if isinstance(x, np.ndarray):
                        x = torch.FloatTensor(x)
                    return self.model(x).cpu().numpy()
            
            if self.background_data is not None:
                self.explainer = shap.DeepExplainer(model_predict, self.background_data)
            else:
                self.explainer = shap.GradientExplainer(model_predict, self.background_data)
    
    def explain_instance(self, instance):
        """Explain a single instance"""
        shap_values = self.explainer.shap_values(instance)
        return shap_values
    
    def explain_batch(self, batch):
        """Explain a batch of instances"""
        shap_values = self.explainer.shap_values(batch)
        return shap_values
    
    def plot_waterfall(self, instance, class_idx=0, max_display=10):
        """Plot waterfall chart for feature importance"""
        shap_values = self.explain_instance(instance)
        if isinstance(shap_values, list):
            shap_values = shap_values[class_idx]
        
        shap.plots.waterfall(shap_values[0], max_display=max_display)
    
    def plot_summary(self, instances, class_names=None, max_display=10):
        """Plot summary of feature importance across instances"""
        shap_values = self.explain_batch(instances)
        
        if isinstance(shap_values, list) and class_names:
            for i, class_name in enumerate(class_names):
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values[i], instances, 
                                 feature_names=None, max_display=max_display,
                                 show=False)
                plt.title(f'SHAP Summary - {class_name}')
                plt.show()
        else:
            shap.summary_plot(shap_values, instances, max_display=max_display)

class LIMEExplainer:
    """LIME-based explanations for images"""
    
    def __init__(self, model, class_names=None):
        if not LIME_AVAILABLE:
            raise ImportError("LIME not available. Install with: pip install lime")
        
        self.model = model
        self.class_names = class_names
        self.explainer = lime_image.LimeImageExplainer()
    
    def _predict_fn(self, images):
        """Prediction function for LIME"""
        batch = torch.FloatTensor(images).permute(0, 3, 1, 2)
        
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def explain_image(self, image, top_labels=5, num_samples=1000, num_features=100000):
        """Explain image prediction using LIME"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)
        
        # Normalize image to [0, 1]
        if image.max() > 1:
            image = image / 255.0
        
        explanation = self.explainer.explain_instance(
            image, 
            self._predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
            num_features=num_features,
            segmentation_fn=SegmentationAlgorithm('quickshift', kernel_size=4,
                                                 max_dist=200, ratio=0.2)
        )
        
        return explanation
    
    def visualize_explanation(self, explanation, positive_only=True, hide_rest=False):
        """Visualize LIME explanation"""
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=positive_only, 
            num_features=10, 
            hide_rest=hide_rest
        )
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(explanation.image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(temp)
        plt.title('Positive Regions' if positive_only else 'Explanation')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(mask, cmap='RdYlBu_r')
        plt.title('Importance Mask')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

class FeatureImportanceAnalyzer:
    """Analyze feature importance for agricultural predictions"""
    
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
    
    def permutation_importance(self, X, y, metric='accuracy', n_repeats=10):
        """Calculate permutation importance"""
        from sklearn.inspection import permutation_importance
        
        if hasattr(self.model, 'predict'):
            result = permutation_importance(self.model, X, y, 
                                          scoring=metric, n_repeats=n_repeats)
            return result
        else:
            logger.warning("Permutation importance requires sklearn-compatible model")
            return None
    
    def plot_feature_importance(self, importance_scores, top_k=20):
        """Plot feature importance"""
        if isinstance(importance_scores, dict):
            # From permutation_importance
            importances = importance_scores['importances_mean']
            std = importance_scores['importances_std']
        else:
            importances = importance_scores
            std = None
        
        # Get top features
        indices = np.argsort(importances)[::-1][:top_k]
        
        plt.figure(figsize=(12, 8))
        
        if self.feature_names:
            labels = [self.feature_names[i] for i in indices]
        else:
            labels = [f'Feature {i}' for i in indices]
        
        y_pos = np.arange(len(labels))
        
        if std is not None:
            plt.barh(y_pos, importances[indices], xerr=std[indices], 
                    capsize=5, alpha=0.7)
        else:
            plt.barh(y_pos, importances[indices], alpha=0.7)
        
        plt.yticks(y_pos, labels)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

class AgriculturalExplainer:
    """Specialized explainer for agricultural predictions"""
    
    def __init__(self, model, class_names=None):
        self.model = model
        self.class_names = class_names or ['Healthy', 'Disease', 'Pest', 'Nutrient Deficiency', 'Water Stress']
        
    def explain_crop_health(self, image, sensor_data=None):
        """Explain crop health prediction"""
        explanations = {}
        
        # Image-based explanation using Grad-CAM
        from .grad_cam import explain_agricultural_prediction
        
        try:
            image_explanation = explain_agricultural_prediction(
                self.model, image, class_names=self.class_names
            )
            explanations['image'] = image_explanation
        except Exception as e:
            logger.warning(f"Image explanation failed: {e}")
        
        # Sensor data explanation using SHAP (if available)
        if sensor_data is not None and SHAP_AVAILABLE:
            try:
                shap_explainer = SHAPExplainer(self.model)
                sensor_explanation = shap_explainer.explain_instance(sensor_data)
                explanations['sensor'] = sensor_explanation
            except Exception as e:
                logger.warning(f"Sensor explanation failed: {e}")
        
        return explanations
    
    def generate_recommendations(self, explanations, prediction):
        """Generate actionable recommendations based on explanations"""
        recommendations = []
        
        predicted_class = prediction.get('predicted_class', 0)
        confidence = prediction.get('confidence', 0.0)
        
        if predicted_class == 0:  # Healthy
            recommendations.append("Crop appears healthy. Continue current care routine.")
        elif predicted_class == 1:  # Disease
            recommendations.extend([
                "Disease detected. Consider applying appropriate fungicide.",
                "Improve air circulation around plants.",
                "Monitor for spread to neighboring plants."
            ])
        elif predicted_class == 2:  # Pest
            recommendations.extend([
                "Pest damage identified. Apply targeted pest control measures.",
                "Check for beneficial insects before treatment.",
                "Consider integrated pest management approaches."
            ])
        elif predicted_class == 3:  # Nutrient Deficiency
            recommendations.extend([
                "Nutrient deficiency detected. Test soil pH and nutrient levels.",
                "Apply appropriate fertilizer based on deficiency type.",
                "Consider slow-release fertilizers for sustained nutrition."
            ])
        elif predicted_class == 4:  # Water Stress
            recommendations.extend([
                "Water stress identified. Adjust irrigation schedule.",
                "Check soil moisture levels regularly.",
                "Consider mulching to retain soil moisture."
            ])
        
        if confidence < 0.7:
            recommendations.append("Low confidence prediction. Consider additional monitoring.")
        
        return recommendations

def main():
    """Test interpretability functions"""
    logger.info("Interpretability module initialized")
    
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available - install with: pip install shap")
    
    if not LIME_AVAILABLE:
        logger.warning("LIME not available - install with: pip install lime")

if __name__ == "__main__":
    main()