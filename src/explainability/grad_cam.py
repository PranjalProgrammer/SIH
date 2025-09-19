"""
Grad-CAM Implementation for Agricultural Image Analysis
Gradient-weighted Class Activation Mapping for model interpretability
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class GradCAM:
    """Grad-CAM implementation for CNN interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
                break
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx].sum()
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam) if np.max(cam) > 0 else cam
        
        return cam
    
    def visualize_cam(self, input_image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """Visualize Grad-CAM overlay on original image"""
        # Resize CAM to match input image
        if isinstance(input_image, torch.Tensor):
            input_image = input_image.cpu().numpy().transpose(1, 2, 0)
            input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
        
        height, width = input_image.shape[:2]
        cam_resized = cv2.resize(cam, (width, height))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Normalize input image
        input_image = np.uint8(255 * input_image)
        
        # Overlay
        overlay = heatmap * alpha + input_image * (1 - alpha)
        overlay = np.uint8(overlay)
        
        return overlay, heatmap
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()

class GuidedBackprop:
    """Guided Backpropagation for detailed feature visualization"""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to modify ReLU gradients"""
        def relu_hook_function(module, grad_in, grad_out):
            if isinstance(module, torch.nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                self.hooks.append(module.register_backward_hook(relu_hook_function))
    
    def generate_gradients(self, input_tensor, class_idx):
        """Generate guided backpropagation gradients"""
        input_tensor.requires_grad_()
        
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        class_score = output[:, class_idx].sum()
        class_score.backward()
        
        return input_tensor.grad.cpu().data.numpy()[0]
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()

class IntegratedGradients:
    """Integrated Gradients for attribution analysis"""
    
    def __init__(self, model):
        self.model = model
    
    def generate_integrated_gradients(self, input_tensor, class_idx, baseline=None, steps=50):
        """Generate integrated gradients"""
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps)
        gradients = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_()
            
            output = self.model(interpolated)
            self.model.zero_grad()
            
            class_score = output[:, class_idx].sum()
            class_score.backward()
            
            gradients.append(interpolated.grad.cpu().data.numpy()[0])
        
        # Average gradients and multiply by input difference
        avg_gradients = np.mean(gradients, axis=0)
        integrated_gradients = (input_tensor.cpu().data.numpy()[0] - baseline.cpu().data.numpy()[0]) * avg_gradients
        
        return integrated_gradients

def explain_agricultural_prediction(model, image, target_layer='features.11', class_names=None):
    """
    Comprehensive explanation of agricultural image prediction
    """
    device = next(model.parameters()).device
    
    # Preprocess image
    if isinstance(image, np.ndarray):
        image_tensor = torch.FloatTensor(image).unsqueeze(0).to(device)
    else:
        image_tensor = image.to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # Generate explanations
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_cam(image_tensor, predicted_class)
    
    # Visualize
    overlay, heatmap = grad_cam.visualize_cam(image_tensor[0], cam)
    
    # Cleanup
    grad_cam.cleanup()
    
    explanation = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'class_name': class_names[predicted_class] if class_names else f'Class {predicted_class}',
        'cam': cam,
        'overlay': overlay,
        'heatmap': heatmap
    }
    
    logger.info(f"Prediction: {explanation['class_name']} (confidence: {confidence:.3f})")
    
    return explanation

def main():
    """Test Grad-CAM functionality"""
    logger.info("Grad-CAM module initialized")

if __name__ == "__main__":
    main()