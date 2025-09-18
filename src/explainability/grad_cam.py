"""
Grad-CAM Implementation for Hyperspectral Analysis

Provides gradient-based class activation mapping for understanding
which spatial regions and spectral bands contribute to predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
import logging
from pathlib import Path
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM for spatial activation mapping."""
    
    def __init__(self, 
                 model: nn.Module,
                 target_layers: List[str],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Target model
            target_layers: Names of layers to analyze
            device: Computation device
        """
        self.model = model.to(device).eval()
        self.device = device
        self.target_layers = target_layers
        
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Find and register hooks for target layers
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                handle_f = module.register_forward_hook(forward_hook(name))
                handle_b = module.register_backward_hook(backward_hook(name))
                self.hooks.extend([handle_f, handle_b])
                logger.info(f"Registered hooks for layer: {name}")
    
    def generate_cam(self, 
                    input_tensor: torch.Tensor,
                    target_class: Optional[int] = None,
                    layer_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Generate class activation maps.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index (None for predicted class)
            layer_name: Specific layer to analyze (None for all)
            
        Returns:
            Dictionary of CAMs for each layer
        """
        self.model.zero_grad()
        
        # Forward pass
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Get model output
        try:
            output = self.model(input_tensor)
            
            # Handle different output formats
            if isinstance(output, tuple):
                logits = output[0]
            elif isinstance(output, dict):
                logits = output['predictions']
            else:
                logits = output
            
            # Get target class
            if target_class is None:
                target_class = torch.argmax(logits, dim=1).item()
            
            # Backward pass for target class
            class_score = logits[0, target_class]
            class_score.backward(retain_graph=True)
            
        except Exception as e:
            logger.error(f"Model forward/backward pass failed: {e}")
            return {}
        
        # Generate CAMs
        cams = {}
        layers_to_process = [layer_name] if layer_name else self.target_layers
        
        for layer in layers_to_process:
            if layer in self.gradients and layer in self.activations:
                cam = self._compute_cam(
                    self.gradients[layer], 
                    self.activations[layer]
                )
                cams[layer] = cam
        
        return cams
    
    def _compute_cam(self, gradients: torch.Tensor, activations: torch.Tensor) -> np.ndarray:
        """Compute class activation map from gradients and activations."""
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def visualize_cam(self, 
                     cam: np.ndarray,
                     original_image: np.ndarray,
                     alpha: float = 0.4) -> np.ndarray:
        """
        Visualize CAM overlaid on original image.
        
        Args:
            cam: Class activation map
            original_image: Original image
            alpha: Overlay transparency
            
        Returns:
            Visualization image
        """
        # Resize CAM to match original image
        if cam.shape != original_image.shape[:2]:
            cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # Ensure original image is in correct format
        if original_image.ndim == 3 and original_image.shape[2] > 3:
            # For hyperspectral, use RGB bands
            rgb_bands = [29, 19, 9]  # Approximate R, G, B band indices
            original_rgb = original_image[:, :, rgb_bands]
        else:
            original_rgb = original_image
        
        # Normalize original image
        if original_rgb.max() <= 1.0:
            original_rgb = (original_rgb * 255).astype(np.uint8)
        
        # Overlay heatmap
        visualization = cv2.addWeighted(original_rgb, 1-alpha, heatmap, alpha, 0)
        
        return visualization
    
    def cleanup(self):
        """Remove hooks and cleanup."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.gradients.clear()
        self.activations.clear()


class SpectralGradCAM:
    """Grad-CAM specialized for spectral analysis."""
    
    def __init__(self, 
                 model: nn.Module,
                 wavelengths: np.ndarray,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Spectral Grad-CAM.
        
        Args:
            model: Target model
            wavelengths: Wavelength array for spectral bands
            device: Computation device
        """
        self.model = model.to(device).eval()
        self.device = device
        self.wavelengths = wavelengths
        
        self.spectral_gradients = None
        self.hook = None
        
        self._register_spectral_hook()
    
    def _register_spectral_hook(self):
        """Register hook for spectral gradients."""
        def spectral_hook(grad):
            self.spectral_gradients = grad.detach()
        
        # Find the input layer or first conv layer
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) >= 3:
                self.hook = param.register_hook(spectral_hook)
                logger.info(f"Registered spectral hook for: {name}")
                break
    
    def generate_spectral_importance(self, 
                                   input_tensor: torch.Tensor,
                                   target_class: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate spectral importance maps.
        
        Args:
            input_tensor: Input hyperspectral tensor
            target_class: Target class for analysis
            
        Returns:
            Dictionary with spectral importance information
        """
        self.model.zero_grad()
        
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        try:
            # Forward pass
            output = self.model(input_tensor)
            
            # Handle different output formats
            if isinstance(output, tuple):
                logits = output[0]
            elif isinstance(output, dict):
                logits = output['predictions']
            else:
                logits = output
            
            # Get target class
            if target_class is None:
                target_class = torch.argmax(logits, dim=1).item()
            
            # Backward pass
            class_score = logits[0, target_class]
            class_score.backward()
            
            # Get input gradients
            input_gradients = input_tensor.grad
            
            # Compute spectral importance
            spectral_importance = self._compute_spectral_importance(
                input_tensor, input_gradients
            )
            
            return {
                'spectral_importance': spectral_importance,
                'wavelengths': self.wavelengths,
                'target_class': target_class,
                'class_score': class_score.item()
            }
            
        except Exception as e:
            logger.error(f"Spectral Grad-CAM failed: {e}")
            return {}
    
    def _compute_spectral_importance(self, 
                                   input_tensor: torch.Tensor,
                                   gradients: torch.Tensor) -> np.ndarray:
        """Compute spectral importance from input and gradients."""
        # Method 1: Gradient * Input (Integrated Gradients style)
        importance = torch.abs(gradients * input_tensor)
        
        # Average across spatial dimensions and batch
        spectral_importance = torch.mean(importance, dim=(0, 1, 2))
        
        return spectral_importance.cpu().numpy()
    
    def plot_spectral_importance(self, 
                                importance_data: Dict[str, np.ndarray],
                                save_path: Optional[str] = None) -> None:
        """
        Plot spectral importance.
        
        Args:
            importance_data: Output from generate_spectral_importance
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return
        
        spectral_importance = importance_data['spectral_importance']
        wavelengths = importance_data['wavelengths']
        
        plt.figure(figsize=(12, 6))
        plt.plot(wavelengths, spectral_importance, linewidth=2)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Importance Score')
        plt.title(f'Spectral Importance for Class {importance_data["target_class"]}')
        plt.grid(True, alpha=0.3)
        
        # Highlight important spectral regions
        threshold = np.percentile(spectral_importance, 90)
        important_regions = spectral_importance > threshold
        plt.fill_between(wavelengths, 0, spectral_importance, 
                        where=important_regions, alpha=0.3, color='red',
                        label='Top 10% Important Bands')
        
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Spectral importance plot saved to {save_path}")
        
        plt.show()
    
    def get_important_bands(self, 
                           importance_data: Dict[str, np.ndarray],
                           top_k: int = 10) -> Dict[str, List]:
        """
        Get the most important spectral bands.
        
        Args:
            importance_data: Output from generate_spectral_importance
            top_k: Number of top bands to return
            
        Returns:
            Dictionary with important band information
        """
        spectral_importance = importance_data['spectral_importance']
        wavelengths = importance_data['wavelengths']
        
        # Get top-k most important bands
        top_indices = np.argsort(spectral_importance)[-top_k:][::-1]
        
        important_bands = {
            'band_indices': top_indices.tolist(),
            'wavelengths': wavelengths[top_indices].tolist(),
            'importance_scores': spectral_importance[top_indices].tolist(),
            'spectral_regions': self._identify_spectral_regions(wavelengths[top_indices])
        }
        
        return important_bands
    
    def _identify_spectral_regions(self, wavelengths: np.ndarray) -> List[str]:
        """Identify spectral regions for given wavelengths."""
        regions = []
        
        for wl in wavelengths:
            if 400 <= wl <= 500:
                regions.append('Blue')
            elif 500 <= wl <= 600:
                regions.append('Green')
            elif 600 <= wl <= 700:
                regions.append('Red')
            elif 700 <= wl <= 800:
                regions.append('Red Edge')
            elif 800 <= wl <= 1300:
                regions.append('NIR')
            elif 1300 <= wl <= 1800:
                regions.append('SWIR1')
            elif 1800 <= wl <= 2500:
                regions.append('SWIR2')
            else:
                regions.append('Other')
        
        return regions
    
    def cleanup(self):
        """Remove hooks and cleanup."""
        if self.hook:
            self.hook.remove()


def create_grad_cam_analyzer(model: nn.Module, 
                           model_type: str = 'cnn',
                           **kwargs) -> Union[GradCAM, SpectralGradCAM]:
    """
    Factory function to create appropriate Grad-CAM analyzer.
    
    Args:
        model: Target model
        model_type: Type of model ('cnn', 'spectral')
        **kwargs: Additional arguments
        
    Returns:
        Grad-CAM analyzer instance
    """
    if model_type == 'spectral':
        return SpectralGradCAM(model, **kwargs)
    else:
        # Default target layers for CNN models
        if 'target_layers' not in kwargs:
            kwargs['target_layers'] = ['conv_layers.4', 'conv_layers.7']  # Last conv layers
        return GradCAM(model, **kwargs)


def main():
    """Demonstrate Grad-CAM functionality."""
    logger.info("Demonstrating Grad-CAM capabilities...")
    
    logger.info("Grad-CAM Features:")
    logger.info("✓ Spatial activation mapping")
    logger.info("✓ Spectral importance analysis")
    logger.info("✓ Multi-layer analysis")
    logger.info("✓ Visualization tools")
    logger.info("✓ Important band identification")
    logger.info("✓ Spectral region analysis")
    
    logger.info("\nGrad-CAM provides insights into:")
    logger.info("• Which spatial regions are most important for classification")
    logger.info("• Which spectral bands contribute most to predictions")
    logger.info("• How different layers focus on different features")
    logger.info("• Model attention patterns across wavelengths")
    
    logger.info("\nGrad-CAM module ready for model interpretation!")


if __name__ == "__main__":
    main()