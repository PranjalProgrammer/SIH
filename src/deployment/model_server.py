"""
Model Server for Precision Agriculture Platform
High-performance model serving with caching and load balancing
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import redis
import pickle
import json
import logging
from datetime import datetime, timedelta
import os
import threading
from queue import Queue
import time

logger = logging.getLogger(__name__)

class ModelCache:
    """Redis-based model caching"""
    
    def __init__(self, redis_host='localhost', redis_port=6379, ttl=3600):
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
            self.redis_client.ping()  # Test connection
            self.ttl = ttl
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.warning(f"Redis cache unavailable: {e}")
            self.redis_client = None
    
    def get(self, key):
        """Get cached result"""
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(key)
            if cached:
                return pickle.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    def set(self, key, value):
        """Cache result"""
        if not self.redis_client:
            return
        
        try:
            serialized = pickle.dumps(value)
            self.redis_client.setex(key, self.ttl, serialized)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, key):
        """Delete cached result"""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")

class ModelPool:
    """Pool of model instances for concurrent serving"""
    
    def __init__(self, model_class, model_path, pool_size=3):
        self.model_class = model_class
        self.model_path = model_path
        self.pool_size = pool_size
        self.models = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize model pool"""
        for i in range(self.pool_size):
            try:
                model = self._load_model()
                self.models.put(model)
                logger.info(f"Model {i+1}/{self.pool_size} loaded into pool")
            except Exception as e:
                logger.error(f"Failed to load model {i+1}: {e}")
    
    def _load_model(self):
        """Load a single model instance"""
        model = self.model_class()
        
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location='cpu')
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        
        model.eval()
        return model
    
    def get_model(self, timeout=30):
        """Get model from pool"""
        try:
            return self.models.get(timeout=timeout)
        except:
            logger.warning("Model pool exhausted, creating new instance")
            return self._load_model()
    
    def return_model(self, model):
        """Return model to pool"""
        try:
            self.models.put_nowait(model)
        except:
            # Pool is full, discard the model
            pass

class BatchProcessor:
    """Batch processing for improved throughput"""
    
    def __init__(self, max_batch_size=8, max_wait_time=0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.batch_queue = Queue()
        self.results = {}
        self.processing = False
        
    def add_request(self, request_id, data):
        """Add request to batch queue"""
        self.batch_queue.put((request_id, data))
        
        if not self.processing:
            self.processing = True
            threading.Thread(target=self._process_batch, daemon=True).start()
    
    def _process_batch(self):
        """Process accumulated batch"""
        batch_data = []
        request_ids = []
        start_time = time.time()
        
        # Collect batch
        while (len(batch_data) < self.max_batch_size and 
               time.time() - start_time < self.max_wait_time):
            
            try:
                request_id, data = self.batch_queue.get(timeout=0.01)
                batch_data.append(data)
                request_ids.append(request_id)
            except:
                if batch_data:  # Process if we have any data
                    break
                continue
        
        if batch_data:
            # Process batch (placeholder)
            batch_results = self._process_batch_data(batch_data)
            
            # Store results
            for req_id, result in zip(request_ids, batch_results):
                self.results[req_id] = result
        
        self.processing = False
    
    def _process_batch_data(self, batch_data):
        """Process batch data (to be implemented by specific models)"""
        # Placeholder implementation
        return [{'processed': True} for _ in batch_data]
    
    def get_result(self, request_id, timeout=5):
        """Get result for request"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in self.results:
                result = self.results.pop(request_id)
                return result
            time.sleep(0.01)
        
        return None

class ModelServer:
    """High-performance model server"""
    
    def __init__(self, model_registry=None, cache_config=None):
        self.model_registry = model_registry or {}
        self.model_pools = {}
        self.cache = ModelCache(**(cache_config or {}))
        self.batch_processors = {}
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize model pools"""
        for model_name, config in self.model_registry.items():
            try:
                pool = ModelPool(
                    model_class=config['class'],
                    model_path=config['path'],
                    pool_size=config.get('pool_size', 3)
                )
                self.model_pools[model_name] = pool
                
                # Initialize batch processor if enabled
                if config.get('batch_processing', False):
                    self.batch_processors[model_name] = BatchProcessor(
                        max_batch_size=config.get('max_batch_size', 8),
                        max_wait_time=config.get('max_wait_time', 0.1)
                    )
                
                logger.info(f"Model server initialized for {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {model_name}: {e}")
    
    def register_model(self, name, model_class, model_path, **config):
        """Register a new model"""
        self.model_registry[name] = {
            'class': model_class,
            'path': model_path,
            **config
        }
        
        # Initialize model pool
        try:
            pool = ModelPool(model_class, model_path, config.get('pool_size', 3))
            self.model_pools[name] = pool
            
            if config.get('batch_processing', False):
                self.batch_processors[name] = BatchProcessor(
                    max_batch_size=config.get('max_batch_size', 8),
                    max_wait_time=config.get('max_wait_time', 0.1)
                )
            
            logger.info(f"Model {name} registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to register model {name}: {e}")
            raise
    
    async def predict(self, model_name, input_data, use_cache=True, **kwargs):
        """Make prediction with caching and pooling"""
        self.stats['requests'] += 1
        
        # Generate cache key
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(model_name, input_data)
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                self.stats['cache_hits'] += 1
                return cached_result
            
            self.stats['cache_misses'] += 1
        
        try:
            # Get prediction
            if model_name in self.batch_processors:
                result = await self._predict_batch(model_name, input_data, **kwargs)
            else:
                result = await self._predict_single(model_name, input_data, **kwargs)
            
            # Cache result
            if use_cache and cache_key:
                self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Prediction error for {model_name}: {e}")
            raise
    
    async def _predict_single(self, model_name, input_data, **kwargs):
        """Single prediction using model pool"""
        if model_name not in self.model_pools:
            raise ValueError(f"Model {model_name} not found")
        
        pool = self.model_pools[model_name]
        model = pool.get_model()
        
        try:
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor, self._run_inference, model, input_data, kwargs
                )
            
            return result
            
        finally:
            pool.return_model(model)
    
    async def _predict_batch(self, model_name, input_data, **kwargs):
        """Batch prediction"""
        if model_name not in self.batch_processors:
            return await self._predict_single(model_name, input_data, **kwargs)
        
        processor = self.batch_processors[model_name]
        request_id = f"{model_name}_{int(time.time() * 1000000)}"
        
        processor.add_request(request_id, (input_data, kwargs))
        
        # Wait for result
        result = processor.get_result(request_id)
        if result is None:
            raise TimeoutError("Batch processing timeout")
        
        return result
    
    def _run_inference(self, model, input_data, kwargs):
        """Run model inference"""
        with torch.no_grad():
            if isinstance(input_data, dict):
                # Multi-input model
                inputs = {k: torch.FloatTensor(v) if isinstance(v, np.ndarray) else v 
                         for k, v in input_data.items()}
                output = model(**inputs)
            else:
                # Single input
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.FloatTensor(input_data)
                else:
                    input_tensor = input_data
                
                output = model(input_tensor)
            
            # Convert output to numpy
            if isinstance(output, torch.Tensor):
                result = output.cpu().numpy()
            elif isinstance(output, (list, tuple)):
                result = [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in output]
            else:
                result = output
            
            return {
                'prediction': result,
                'timestamp': datetime.now().isoformat(),
                'model': type(model).__name__
            }
    
    def _generate_cache_key(self, model_name, input_data):
        """Generate cache key for input"""
        if isinstance(input_data, np.ndarray):
            data_hash = hash(input_data.tobytes())
        elif isinstance(input_data, dict):
            # Hash each array in the dict
            hash_components = []
            for k, v in sorted(input_data.items()):
                if isinstance(v, np.ndarray):
                    hash_components.append(f"{k}:{hash(v.tobytes())}")
                else:
                    hash_components.append(f"{k}:{hash(str(v))}")
            data_hash = hash(":".join(hash_components))
        else:
            data_hash = hash(str(input_data))
        
        return f"{model_name}:{data_hash}"
    
    def get_stats(self):
        """Get server statistics"""
        return {
            **self.stats,
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['requests'], 1),
            'models_loaded': len(self.model_pools),
            'timestamp': datetime.now().isoformat()
        }
    
    def health_check(self):
        """Health check for server"""
        status = {
            'status': 'healthy',
            'models': {},
            'cache': 'unavailable' if not self.cache.redis_client else 'available',
            'timestamp': datetime.now().isoformat()
        }
        
        for name, pool in self.model_pools.items():
            status['models'][name] = {
                'available': not pool.models.empty(),
                'pool_size': pool.pool_size
            }
        
        return status

def create_model_server(config_path=None):
    """Create model server from configuration"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model_registry = config.get('models', {})
        cache_config = config.get('cache', {})
    else:
        # Default configuration
        model_registry = {}
        cache_config = {}
    
    return ModelServer(model_registry, cache_config)

def main():
    """Test model server"""
    server = create_model_server()
    logger.info("Model server initialized")
    print(server.health_check())

if __name__ == "__main__":
    main()