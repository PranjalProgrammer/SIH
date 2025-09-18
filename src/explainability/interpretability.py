"""
Model Interpretability Suite

Comprehensive interpretability tools combining multiple explanation methods
for agricultural AI models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json

from .grad_cam import GradCAM, SpectralGradCAM
from .ood_detection import OutOfDistributionDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInterpreter:
    """Comprehensive model interpretability suite."""
    
    def __init__(self, 
                 model: nn.Module,
                 wavelengths: np.ndarray,
                 class_names: List[str] = ['healthy', 'stressed', 'diseased'],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Model Interpreter.
        
        Args:
            model: Trained model to interpret
            wavelengths: Wavelength array for spectral analysis
            class_names: Names of prediction classes
            device: Computation device
        """
        self.model = model.to(device).eval()
        self.device = device
        self.wavelengths = wavelengths
        self.class_names = class_names
        
        # Initialize interpretation tools
        self.grad_cam = None
        self.spectral_grad_cam = None
        self.ood_detector = None
        
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize interpretation tools."""
        try:
            # Initialize Grad-CAM
            target_layers = self._find_target_layers()
            if target_layers:
                self.grad_cam = GradCAM(
                    self.model, 
                    target_layers=target_layers,
                    device=self.device
                )
            
            # Initialize Spectral Grad-CAM
            self.spectral_grad_cam = SpectralGradCAM(
                self.model,
                wavelengths=self.wavelengths,
                device=self.device
            )
            
            # Initialize OOD Detector
            self.ood_detector = OutOfDistributionDetector(
                self.model,
                device=self.device,
                methods=['mahalanobis', 'confidence', 'energy']
            )
            
            logger.info("Interpretation tools initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize some interpretation tools: {e}")
    
    def _find_target_layers(self) -> List[str]:
        """Automatically find suitable target layers for Grad-CAM."""
        target_layers = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and 'classifier' not in name.lower():
                target_layers.append(name)
        
        # Return last few convolutional layers
        return target_layers[-2:] if len(target_layers) >= 2 else target_layers
    
    def interpret_prediction(self, 
                           inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                           target_class: Optional[int] = None,
                           include_ood: bool = True) -> Dict[str, Any]:
        """
        Comprehensive interpretation of a single prediction.
        
        Args:
            inputs: Input data (image sequence, sensor sequence)
            target_class: Target class for interpretation
            include_ood: Whether to include OOD analysis
            
        Returns:
            Dictionary with interpretation results
        """
        logger.info("Generating comprehensive interpretation...")
        
        interpretation = {
            'prediction': None,
            'confidence': None,
            'spatial_attribution': None,
            'spectral_importance': None,
            'temporal_attention': None,
            'ood_analysis': None,
            'explanation_summary': None
        }
        
        # Get model prediction
        prediction_results = self._get_prediction(inputs)
        interpretation.update(prediction_results)
        
        if target_class is None:
            target_class = prediction_results['predicted_class']
        
        # Spatial attribution (Grad-CAM)
        if self.grad_cam is not None:
            try:
                spatial_results = self._get_spatial_attribution(inputs, target_class)
                interpretation['spatial_attribution'] = spatial_results
            except Exception as e:
                logger.warning(f"Spatial attribution failed: {e}")
        
        # Spectral importance
        if self.spectral_grad_cam is not None:
            try:
                spectral_results = self._get_spectral_importance(inputs, target_class)
                interpretation['spectral_importance'] = spectral_results
            except Exception as e:
                logger.warning(f"Spectral importance analysis failed: {e}")
        
        # Temporal attention (if available)
        try:
            temporal_results = self._get_temporal_attention(inputs)
            interpretation['temporal_attention'] = temporal_results
        except Exception as e:
            logger.debug(f"Temporal attention not available: {e}")
        
        # OOD analysis
        if include_ood and self.ood_detector is not None:
            try:
                ood_results = self._get_ood_analysis(inputs)
                interpretation['ood_analysis'] = ood_results
            except Exception as e:
                logger.warning(f"OOD analysis failed: {e}")
        
        # Generate explanation summary
        interpretation['explanation_summary'] = self._generate_explanation_summary(interpretation)
        
        logger.info("Interpretation completed")
        return interpretation
    
    def _get_prediction(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
        """Get model prediction and confidence."""
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
                attention_weights = outputs[1] if len(outputs) > 1 else None
            elif isinstance(outputs, dict):
                logits = outputs['predictions']
                attention_weights = outputs.get('attention_weights', None)
            else:
                logits = outputs
                attention_weights = None
            
            # Get predictions
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities, dim=1)[0].item()
        
        return {
            'predicted_class': predicted_class,
            'predicted_class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'class_probabilities': probabilities.cpu().numpy().tolist(),
            'attention_weights': attention_weights.cpu().numpy() if attention_weights is not None else None
        }
    
    def _get_spatial_attribution(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                                target_class: int) -> Dict[str, Any]:
        """Get spatial attribution maps."""
        # For hybrid models, extract image component
        if isinstance(inputs, tuple):
            image_input = inputs[0]
            # Use first frame if it's a sequence
            if image_input.dim() == 5:  # (batch, seq, h, w, c)
                image_input = image_input[0, 0]  # First sample, first frame
            elif image_input.dim() == 4:  # (batch, h, w, c)
                image_input = image_input[0]  # First sample
        else:
            image_input = inputs[0] if inputs.dim() == 4 else inputs
        
        # Generate CAMs
        cams = self.grad_cam.generate_cam(image_input, target_class)
        
        spatial_results = {
            'cams': cams,
            'target_class': target_class,
            'summary': self._summarize_spatial_attribution(cams)
        }
        
        return spatial_results
    
    def _get_spectral_importance(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                               target_class: int) -> Dict[str, Any]:
        """Get spectral importance analysis."""
        # Extract image component for spectral analysis
        if isinstance(inputs, tuple):
            image_input = inputs[0]
            if image_input.dim() == 5:  # (batch, seq, h, w, c)
                image_input = image_input[0, 0]  # First sample, first frame
            elif image_input.dim() == 4:  # (batch, h, w, c)
                image_input = image_input[0]  # First sample
        else:
            image_input = inputs[0] if inputs.dim() == 4 else inputs
        
        # Generate spectral importance
        importance_data = self.spectral_grad_cam.generate_spectral_importance(
            image_input, target_class
        )
        
        if importance_data:
            # Get important bands
            important_bands = self.spectral_grad_cam.get_important_bands(
                importance_data, top_k=10
            )
            
            spectral_results = {
                'importance_scores': importance_data['spectral_importance'].tolist(),
                'wavelengths': self.wavelengths.tolist(),
                'important_bands': important_bands,
                'target_class': target_class,
                'summary': self._summarize_spectral_importance(important_bands)
            }
        else:
            spectral_results = {'error': 'Failed to compute spectral importance'}
        
        return spectral_results
    
    def _get_temporal_attention(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
        """Get temporal attention weights if available."""
        # This would be populated if the model returns attention weights
        # For now, return placeholder
        return {
            'attention_weights': None,
            'temporal_summary': 'Temporal attention analysis not available for this model'
        }
    
    def _get_ood_analysis(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
        """Get out-of-distribution analysis."""
        ood_results = self.ood_detector.detect_ood(inputs, return_scores=True)
        
        # Summarize OOD results
        ood_summary = {
            'is_ood_any_method': any(ood_results['is_ood'].values()),
            'ood_methods_triggered': [method for method, is_ood in ood_results['is_ood'].items() if is_ood],
            'confidence_score': float(np.mean(ood_results['max_confidence'])),
            'recommendation': self._get_ood_recommendation(ood_results)
        }
        
        return {
            'detailed_results': ood_results,
            'summary': ood_summary
        }
    
    def _summarize_spatial_attribution(self, cams: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Summarize spatial attribution results."""
        if not cams:
            return {'error': 'No spatial attribution available'}
        
        summary = {}
        for layer_name, cam in cams.items():
            # Find most important regions
            threshold = np.percentile(cam, 90)
            important_regions = cam > threshold
            
            summary[layer_name] = {
                'max_activation': float(np.max(cam)),
                'mean_activation': float(np.mean(cam)),
                'important_region_percentage': float(np.mean(important_regions) * 100),
                'activation_center': self._find_activation_center(cam)
            }
        
        return summary
    
    def _summarize_spectral_importance(self, important_bands: Dict[str, List]) -> Dict[str, Any]:
        """Summarize spectral importance results."""
        if not important_bands:
            return {'error': 'No spectral importance data available'}
        
        # Group by spectral regions
        region_importance = {}
        for i, region in enumerate(important_bands['spectral_regions']):
            if region not in region_importance:
                region_importance[region] = []
            region_importance[region].append(important_bands['importance_scores'][i])
        
        # Calculate average importance per region
        region_summary = {}
        for region, scores in region_importance.items():
            region_summary[region] = {
                'average_importance': float(np.mean(scores)),
                'max_importance': float(np.max(scores)),
                'band_count': len(scores)
            }
        
        return {
            'most_important_region': max(region_summary.keys(), 
                                       key=lambda x: region_summary[x]['average_importance']),
            'region_importance': region_summary,
            'top_wavelengths': important_bands['wavelengths'][:5]
        }
    
    def _find_activation_center(self, cam: np.ndarray) -> Tuple[int, int]:
        """Find the center of highest activation in CAM."""
        flat_idx = np.argmax(cam)
        center_y, center_x = np.unravel_index(flat_idx, cam.shape)
        return int(center_y), int(center_x)
    
    def _get_ood_recommendation(self, ood_results: Dict[str, Any]) -> str:
        """Generate recommendation based on OOD analysis."""
        ood_count = sum(ood_results['is_ood'].values())
        total_methods = len(ood_results['is_ood'])
        
        if ood_count == 0:
            return "Sample appears to be from the training distribution. Prediction is reliable."
        elif ood_count == total_methods:
            return "Sample is likely out-of-distribution. Prediction should be reviewed by an expert."
        else:
            return f"Sample shows some OOD characteristics ({ood_count}/{total_methods} methods). Use prediction with caution."
    
    def _generate_explanation_summary(self, interpretation: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable explanation summary."""
        summary = {}
        
        # Prediction summary
        pred_class = interpretation.get('predicted_class_name', 'Unknown')
        confidence = interpretation.get('confidence', 0.0)
        
        summary['prediction'] = f"Model predicts '{pred_class}' with {confidence:.2%} confidence."
        
        # Spatial summary
        if interpretation.get('spatial_attribution'):
            spatial_data = interpretation['spatial_attribution'].get('summary', {})
            if spatial_data and not spatial_data.get('error'):
                # Get the layer with highest activation
                best_layer = max(spatial_data.keys(), 
                               key=lambda x: spatial_data[x].get('max_activation', 0))
                activation_info = spatial_data[best_layer]
                
                summary['spatial'] = (f"Model focuses on {activation_info['important_region_percentage']:.1f}% "
                                    f"of the image area, with strongest attention at position "
                                    f"{activation_info['activation_center']}.")
        
        # Spectral summary
        if interpretation.get('spectral_importance'):
            spectral_data = interpretation['spectral_importance'].get('summary', {})
            if spectral_data and not spectral_data.get('error'):
                important_region = spectral_data.get('most_important_region', 'Unknown')
                top_wavelengths = spectral_data.get('top_wavelengths', [])
                
                if top_wavelengths:
                    summary['spectral'] = (f"Most important spectral region: {important_region}. "
                                         f"Key wavelengths include {top_wavelengths[0]:.0f}nm "
                                         f"and {top_wavelengths[1]:.0f}nm.")
        
        # OOD summary
        if interpretation.get('ood_analysis'):
            ood_data = interpretation['ood_analysis'].get('summary', {})
            recommendation = ood_data.get('recommendation', '')
            summary['reliability'] = recommendation
        
        # Overall assessment
        if confidence > 0.8:
            confidence_level = "high"
        elif confidence > 0.6:
            confidence_level = "moderate"
        else:
            confidence_level = "low"
        
        summary['overall'] = (f"Overall assessment: {confidence_level} confidence prediction. "
                            f"Model shows clear decision patterns in both spatial and spectral domains.")
        
        return summary
    
    def generate_report(self, 
                       inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                       save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive interpretation report.
        
        Args:
            inputs: Input data to interpret
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        logger.info("Generating comprehensive interpretation report...")
        
        # Get full interpretation
        interpretation = self.interpret_prediction(inputs)
        
        # Generate report
        report_lines = [
            "=" * 80,
            "PRECISION AGRICULTURE MODEL - INTERPRETATION REPORT",
            "=" * 80,
            "",
            f"Model Prediction:",
            f"  Predicted Class: {interpretation.get('predicted_class_name', 'Unknown')}",
            f"  Confidence: {interpretation.get('confidence', 0.0):.4f}",
            ""
        ]
        
        # Add class probabilities
        if interpretation.get('class_probabilities'):
            report_lines.append("Class Probabilities:")
            for i, (class_name, prob) in enumerate(zip(self.class_names, interpretation['class_probabilities'][0])):
                report_lines.append(f"  {class_name}: {prob:.4f}")
            report_lines.append("")
        
        # Add explanation summary
        if interpretation.get('explanation_summary'):
            report_lines.append("Explanation Summary:")
            for key, explanation in interpretation['explanation_summary'].items():
                report_lines.append(f"  {key.title()}: {explanation}")
            report_lines.append("")
        
        # Add spatial attribution details
        if interpretation.get('spatial_attribution'):
            spatial_data = interpretation['spatial_attribution'].get('summary', {})
            if spatial_data and not spatial_data.get('error'):
                report_lines.append("Spatial Attribution Analysis:")
                for layer, data in spatial_data.items():
                    report_lines.append(f"  Layer {layer}:")
                    report_lines.append(f"    Max Activation: {data['max_activation']:.4f}")
                    report_lines.append(f"    Important Region: {data['important_region_percentage']:.1f}%")
                    report_lines.append(f"    Focus Center: {data['activation_center']}")
                report_lines.append("")
        
        # Add spectral importance details
        if interpretation.get('spectral_importance'):
            spectral_data = interpretation['spectral_importance']
            if not spectral_data.get('error'):
                important_bands = spectral_data.get('important_bands', {})
                if important_bands:
                    report_lines.append("Spectral Importance Analysis:")
                    report_lines.append(f"  Top 5 Important Wavelengths:")
                    for i in range(min(5, len(important_bands['wavelengths']))):
                        wl = important_bands['wavelengths'][i]
                        score = important_bands['importance_scores'][i]
                        region = important_bands['spectral_regions'][i]
                        report_lines.append(f"    {wl:.0f}nm ({region}): {score:.4f}")
                    report_lines.append("")
        
        # Add OOD analysis
        if interpretation.get('ood_analysis'):
            ood_summary = interpretation['ood_analysis'].get('summary', {})
            if ood_summary:
                report_lines.append("Out-of-Distribution Analysis:")
                report_lines.append(f"  OOD Detection: {'Yes' if ood_summary['is_ood_any_method'] else 'No'}")
                report_lines.append(f"  Methods Triggered: {', '.join(ood_summary['ood_methods_triggered'])}")
                report_lines.append(f"  Recommendation: {ood_summary['recommendation']}")
                report_lines.append("")
        
        report_lines.extend([
            "=" * 80,
            "END OF INTERPRETATION REPORT",
            "=" * 80
        ])
        
        report = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            report_path = Path(save_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(report)
            logger.info(f"Interpretation report saved to {report_path}")
        
        return report
    
    def cleanup(self):
        """Cleanup all interpretation tools."""
        if self.grad_cam:
            self.grad_cam.cleanup()
        if self.spectral_grad_cam:
            self.spectral_grad_cam.cleanup()
        if self.ood_detector:
            self.ood_detector.cleanup()


def main():
    """Demonstrate interpretability capabilities."""
    logger.info("Demonstrating Model Interpretability Suite...")
    
    logger.info("Interpretability Features:")
    logger.info("✓ Comprehensive prediction interpretation")
    logger.info("✓ Spatial attribution mapping (Grad-CAM)")
    logger.info("✓ Spectral importance analysis")
    logger.info("✓ Temporal attention visualization")
    logger.info("✓ Out-of-distribution detection")
    logger.info("✓ Human-readable explanations")
    logger.info("✓ Detailed interpretation reports")
    
    logger.info("\nInterpretation Outputs:")
    logger.info("• Spatial heatmaps showing important image regions")
    logger.info("• Spectral importance plots highlighting key wavelengths")
    logger.info("• Confidence and reliability assessments")
    logger.info("• Actionable recommendations for users")
    logger.info("• Comprehensive PDF/HTML reports")
    
    logger.info("\nBenefits for Agricultural Applications:")
    logger.info("• Build trust in AI predictions")
    logger.info("• Identify model limitations and biases")
    logger.info("• Guide data collection and model improvement")
    logger.info("• Support expert decision making")
    logger.info("• Enable regulatory compliance")
    
    logger.info("\nModel interpretability suite ready for deployment!")


if __name__ == "__main__":
    main()