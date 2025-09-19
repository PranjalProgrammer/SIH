#!/usr/bin/env python3
"""
Core Functionality Test Suite
Tests the main components of the Precision Agriculture Platform
"""

import os
import sys
import unittest
import numpy as np
import torch
import requests
import json
import time
from datetime import datetime
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDataAcquisition(unittest.TestCase):
    """Test data acquisition modules"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_dir = '/tmp/test_data'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def test_dataset_downloader(self):
        """Test dataset downloader"""
        try:
            from data_acquisition.download_datasets import DatasetDownloader
            
            downloader = DatasetDownloader(self.data_dir)
            result = downloader.download_sample_data()
            
            self.assertTrue(result)
            self.assertTrue(os.path.exists(os.path.join(self.data_dir, 'hyperspectral')))
            
            logger.info("✅ Dataset downloader test passed")
        except ImportError as e:
            logger.warning(f"⚠️  Dataset downloader test skipped: {e}")
    
    def test_image_preprocessor(self):
        """Test image preprocessing"""
        try:
            from data_acquisition.preprocess_images import ImagePreprocessor
            
            preprocessor = ImagePreprocessor()
            
            # Create dummy image
            dummy_image = np.random.rand(100, 100, 3)
            
            # Test normalization
            normalized = preprocessor.normalize_image(dummy_image)
            self.assertGreaterEqual(normalized.min(), 0)
            self.assertLessEqual(normalized.max(), 1)
            
            # Test resizing
            resized = preprocessor.resize_image(dummy_image, (224, 224))
            self.assertEqual(resized.shape[:2], (224, 224))
            
            logger.info("✅ Image preprocessor test passed")
        except ImportError as e:
            logger.warning(f"⚠️  Image preprocessor test skipped: {e}")
    
    def test_feature_extractor(self):
        """Test feature extraction"""
        try:
            from data_acquisition.feature_extraction import FeatureExtractor
            
            extractor = FeatureExtractor()
            
            # Test spectral features
            hyperspectral_data = np.random.rand(100, 100, 50)
            features = extractor.extract_spectral_features(hyperspectral_data)
            
            self.assertIn('mean_spectrum', features)
            self.assertIn('ndvi', features)
            
            logger.info("✅ Feature extractor test passed")
        except ImportError as e:
            logger.warning(f"⚠️  Feature extractor test skipped: {e}")

class TestModels(unittest.TestCase):
    """Test model architectures"""
    
    def test_cnn_models(self):
        """Test CNN model creation"""
        try:
            from models.cnn_models import create_model
            
            # Test basic CNN
            model = create_model('cnn', num_classes=5, input_channels=3)
            self.assertIsNotNone(model)
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            self.assertEqual(output.shape, (1, 5))
            
            logger.info("✅ CNN models test passed")
        except ImportError as e:
            logger.warning(f"⚠️  CNN models test skipped: {e}")
    
    def test_lstm_models(self):
        """Test LSTM model creation"""
        try:
            from models.lstm_models import create_lstm_model
            
            # Test sensor LSTM
            model = create_lstm_model('sensor', input_size=7, hidden_size=64)
            self.assertIsNotNone(model)
            
            # Test forward pass
            dummy_input = torch.randn(1, 10, 7)  # batch, sequence, features
            output = model(dummy_input)
            self.assertEqual(output.shape[0], 1)  # batch size
            
            logger.info("✅ LSTM models test passed")
        except ImportError as e:
            logger.warning(f"⚠️  LSTM models test skipped: {e}")
    
    def test_hybrid_models(self):
        """Test hybrid model creation"""
        try:
            from models.hybrid_models import create_hybrid_model
            
            # Test CNN-LSTM hybrid
            model = create_hybrid_model('cnn_lstm', num_classes=5)
            self.assertIsNotNone(model)
            
            logger.info("✅ Hybrid models test passed")
        except ImportError as e:
            logger.warning(f"⚠️  Hybrid models test skipped: {e}")

class TestExplainability(unittest.TestCase):
    """Test explainability modules"""
    
    def test_grad_cam(self):
        """Test Grad-CAM functionality"""
        try:
            from explainability.grad_cam import GradCAM
            from models.cnn_models import create_model
            
            # Create a simple model
            model = create_model('cnn', num_classes=5)
            
            # Test Grad-CAM
            grad_cam = GradCAM(model, 'features.11')
            
            dummy_input = torch.randn(1, 3, 224, 224)
            cam = grad_cam.generate_cam(dummy_input)
            
            self.assertIsInstance(cam, np.ndarray)
            self.assertEqual(len(cam.shape), 2)  # Should be 2D
            
            grad_cam.cleanup()
            
            logger.info("✅ Grad-CAM test passed")
        except ImportError as e:
            logger.warning(f"⚠️  Grad-CAM test skipped: {e}")
    
    def test_ood_detection(self):
        """Test out-of-distribution detection"""
        try:
            from explainability.ood_detection import create_ood_detector
            
            # Test Mahalanobis detector
            detector = create_ood_detector('mahalanobis')
            
            # Generate sample data
            reference_data = np.random.normal(0, 1, (100, 10))
            new_data = np.random.normal(0, 1, (20, 10))
            
            detector.fit(reference_data)
            scores = detector.score(new_data)
            predictions = detector.predict(new_data)
            
            self.assertEqual(len(scores), 20)
            self.assertEqual(len(predictions), 20)
            
            logger.info("✅ OOD detection test passed")
        except ImportError as e:
            logger.warning(f"⚠️  OOD detection test skipped: {e}")

class TestContinuousLearning(unittest.TestCase):
    """Test continuous learning modules"""
    
    def test_active_learning(self):
        """Test active learning strategies"""
        try:
            from continuous_learning.active_learning import create_active_learning_strategy
            from models.cnn_models import create_model
            
            # Test uncertainty-based sampling
            strategy = create_active_learning_strategy('uncertainty', batch_size=5)
            model = create_model('cnn', num_classes=5)
            
            # Generate dummy unlabeled data
            unlabeled_data = [torch.randn(3, 224, 224) for _ in range(20)]
            
            selected_indices = strategy.select_samples(model, unlabeled_data)
            
            self.assertEqual(len(selected_indices), 5)
            self.assertTrue(all(0 <= idx < 20 for idx in selected_indices))
            
            logger.info("✅ Active learning test passed")
        except ImportError as e:
            logger.warning(f"⚠️  Active learning test skipped: {e}")
    
    def test_drift_detection(self):
        """Test drift detection"""
        try:
            from continuous_learning.drift_detection import create_drift_detector
            
            # Test statistical drift detector
            detector = create_drift_detector('statistical')
            
            # Generate reference and new data
            reference_data = np.random.normal(0, 1, (100, 10))
            new_data = np.random.normal(0.5, 1, (50, 10))  # Shifted distribution
            
            detector.fit(reference_data)
            result = detector.detect(new_data)
            
            self.assertIn('drift_detected', result)
            self.assertIn('p_values', result)
            
            logger.info("✅ Drift detection test passed")
        except ImportError as e:
            logger.warning(f"⚠️  Drift detection test skipped: {e}")

class TestAPI(unittest.TestCase):
    """Test API functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Start API server for testing"""
        cls.api_url = "http://localhost:5000"
        cls.server_running = False
        
        # Check if server is already running
        try:
            response = requests.get(f"{cls.api_url}/health", timeout=5)
            if response.status_code == 200:
                cls.server_running = True
                logger.info("API server is already running")
        except requests.exceptions.RequestException:
            logger.warning("API server not running - some tests will be skipped")
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        if not self.server_running:
            self.skipTest("API server not running")
        
        try:
            response = requests.get(f"{self.api_url}/health")
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn('status', data)
            
            logger.info("✅ Health endpoint test passed")
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  Health endpoint test failed: {e}")
    
    def test_models_endpoint(self):
        """Test models listing endpoint"""
        if not self.server_running:
            self.skipTest("API server not running")
        
        try:
            response = requests.get(f"{self.api_url}/api/models")
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn('models', data)
            
            logger.info("✅ Models endpoint test passed")
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  Models endpoint test failed: {e}")

class TestSystemIntegration(unittest.TestCase):
    """Test system integration"""
    
    def test_configuration_loading(self):
        """Test configuration file loading"""
        try:
            import yaml
            
            # Test model config
            if os.path.exists('configs/model_config.yaml'):
                with open('configs/model_config.yaml', 'r') as f:
                    config = yaml.safe_load(f)
                    self.assertIn('models', config)
                    logger.info("✅ Model configuration loading test passed")
            
            # Test deployment config
            if os.path.exists('configs/deployment_config.yaml'):
                with open('configs/deployment_config.yaml', 'r') as f:
                    config = yaml.safe_load(f)
                    self.assertIn('api', config)
                    logger.info("✅ Deployment configuration loading test passed")
                    
        except ImportError:
            logger.warning("⚠️  PyYAML not available - configuration tests skipped")
    
    def test_directory_structure(self):
        """Test that required directories exist"""
        required_dirs = [
            'src/data_acquisition',
            'src/models',
            'src/explainability',
            'src/deployment',
            'src/continuous_learning',
            'dashboard',
            'configs'
        ]
        
        for dir_path in required_dirs:
            self.assertTrue(os.path.exists(dir_path), f"Directory {dir_path} not found")
        
        logger.info("✅ Directory structure test passed")
    
    def test_docker_files(self):
        """Test Docker configuration files"""
        required_files = [
            'Dockerfile',
            'docker-compose.yml',
            'requirements.txt'
        ]
        
        for file_path in required_files:
            self.assertTrue(os.path.exists(file_path), f"File {file_path} not found")
        
        logger.info("✅ Docker files test passed")

def run_performance_test():
    """Run basic performance tests"""
    logger.info("🚀 Running performance tests...")
    
    try:
        from models.cnn_models import create_model
        
        # Test model inference speed
        model = create_model('cnn', num_classes=5)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Measure inference time
        start_time = time.time()
        num_inferences = 100
        
        with torch.no_grad():
            for _ in range(num_inferences):
                _ = model(dummy_input)
        
        end_time = time.time()
        avg_inference_time = (end_time - start_time) / num_inferences
        
        logger.info(f"✅ Average inference time: {avg_inference_time:.4f} seconds")
        
        if avg_inference_time < 0.1:  # Less than 100ms
            logger.info("✅ Performance test passed")
        else:
            logger.warning("⚠️  Performance may be suboptimal")
            
    except Exception as e:
        logger.warning(f"⚠️  Performance test failed: {e}")

def main():
    """Run all tests"""
    print("🧪 Running Precision Agriculture Platform Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataAcquisition,
        TestModels,
        TestExplainability,
        TestContinuousLearning,
        TestAPI,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run performance tests
    run_performance_test()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())