"""
FastAPI REST API for Precision Agriculture Platform

Provides endpoints for crop health prediction, model interpretation,
and system monitoring.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import torch
import numpy as np
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
import tempfile
import os

from .model_server import ModelServer
from .monitoring import MetricsCollector, HealthChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for crop health prediction."""
    image_data: List[List[List[List[float]]]] = Field(
        description="Hyperspectral image sequence (seq_len, height, width, bands)"
    )
    sensor_data: List[List[float]] = Field(
        description="Sensor data sequence (seq_len, features)"
    )
    sequence_length: Optional[int] = Field(
        default=None, description="Actual sequence length"
    )
    include_interpretation: bool = Field(
        default=False, description="Include model interpretation"
    )
    include_ood: bool = Field(
        default=True, description="Include out-of-distribution analysis"
    )

class PredictionResponse(BaseModel):
    """Response model for crop health prediction."""
    prediction: Dict[str, Any]
    interpretation: Optional[Dict[str, Any]] = None
    processing_time: float
    timestamp: str
    model_version: str

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    samples: List[PredictionRequest]
    batch_size: int = Field(default=32, description="Batch processing size")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]

class MetricsResponse(BaseModel):
    """Metrics response model."""
    predictions_count: int
    average_processing_time: float
    error_rate: float
    uptime: float
    last_updated: str

# Global instances
model_server: Optional[ModelServer] = None
metrics_collector: Optional[MetricsCollector] = None
health_checker: Optional[HealthChecker] = None

def create_app(
    model_path: str = "/workspace/models/best_model.pth",
    config_path: str = "/workspace/configs/deployment_config.yaml"
) -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Precision Agriculture Platform API",
        description="AI-powered crop health monitoring and prediction system",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize global components
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        global model_server, metrics_collector, health_checker
        
        try:
            # Initialize model server
            logger.info("Initializing model server...")
            model_server = ModelServer(model_path=model_path)
            await model_server.initialize()
            
            # Initialize monitoring
            logger.info("Initializing monitoring services...")
            metrics_collector = MetricsCollector()
            health_checker = HealthChecker(model_server)
            
            logger.info("API services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize API services: {e}")
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        global model_server, metrics_collector, health_checker
        
        if model_server:
            await model_server.cleanup()
        
        logger.info("API services shut down")
    
    return app

# Create app instance
app = create_app()

# Dependency for getting model server
async def get_model_server() -> ModelServer:
    """Dependency to get model server instance."""
    if model_server is None:
        raise HTTPException(
            status_code=503, 
            detail="Model server not initialized"
        )
    return model_server

# Dependency for getting metrics collector
async def get_metrics_collector() -> MetricsCollector:
    """Dependency to get metrics collector instance."""
    if metrics_collector is None:
        raise HTTPException(
            status_code=503,
            detail="Metrics collector not initialized"
        )
    return metrics_collector

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Precision Agriculture Platform API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global health_checker
    
    if health_checker is None:
        raise HTTPException(status_code=503, detail="Health checker not initialized")
    
    try:
        health_status = await health_checker.get_health_status()
        return HealthResponse(**health_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    metrics: MetricsCollector = Depends(get_metrics_collector)
):
    """Get system metrics."""
    try:
        metrics_data = await metrics.get_metrics()
        return MetricsResponse(**metrics_data)
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.post("/predict", response_model=PredictionResponse)
async def predict_crop_health(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    server: ModelServer = Depends(get_model_server),
    metrics: MetricsCollector = Depends(get_metrics_collector)
):
    """Predict crop health from hyperspectral and sensor data."""
    start_time = datetime.now()
    
    try:
        # Convert input data to tensors
        image_tensor = torch.tensor(request.image_data, dtype=torch.float32)
        sensor_tensor = torch.tensor(request.sensor_data, dtype=torch.float32)
        
        # Add batch dimension if needed
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.unsqueeze(0)
        if sensor_tensor.dim() == 2:
            sensor_tensor = sensor_tensor.unsqueeze(0)
        
        # Make prediction
        prediction_result = await server.predict(
            image_sequence=image_tensor,
            sensor_sequence=sensor_tensor,
            sequence_length=request.sequence_length,
            include_interpretation=request.include_interpretation,
            include_ood=request.include_ood
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update metrics in background
        background_tasks.add_task(
            metrics.record_prediction,
            processing_time=processing_time,
            success=True
        )
        
        response = PredictionResponse(
            prediction=prediction_result["prediction"],
            interpretation=prediction_result.get("interpretation"),
            processing_time=processing_time,
            timestamp=start_time.isoformat(),
            model_version=server.model_version
        )
        
        return response
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Record error in metrics
        background_tasks.add_task(
            metrics.record_prediction,
            processing_time=processing_time,
            success=False
        )
        
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    server: ModelServer = Depends(get_model_server),
    metrics: MetricsCollector = Depends(get_metrics_collector)
):
    """Process batch predictions."""
    start_time = datetime.now()
    results = []
    
    try:
        # Process samples in batches
        for i in range(0, len(request.samples), request.batch_size):
            batch_samples = request.samples[i:i + request.batch_size]
            batch_results = []
            
            for sample in batch_samples:
                # Convert to tensors
                image_tensor = torch.tensor(sample.image_data, dtype=torch.float32)
                sensor_tensor = torch.tensor(sample.sensor_data, dtype=torch.float32)
                
                if image_tensor.dim() == 4:
                    image_tensor = image_tensor.unsqueeze(0)
                if sensor_tensor.dim() == 2:
                    sensor_tensor = sensor_tensor.unsqueeze(0)
                
                # Make prediction
                prediction_result = await server.predict(
                    image_sequence=image_tensor,
                    sensor_sequence=sensor_tensor,
                    sequence_length=sample.sequence_length,
                    include_interpretation=sample.include_interpretation,
                    include_ood=sample.include_ood
                )
                
                batch_results.append(prediction_result)
            
            results.extend(batch_results)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update metrics
        background_tasks.add_task(
            metrics.record_batch_prediction,
            batch_size=len(request.samples),
            processing_time=processing_time,
            success=True
        )
        
        return {
            "results": results,
            "batch_size": len(request.samples),
            "processing_time": processing_time,
            "timestamp": start_time.isoformat()
        }
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        
        background_tasks.add_task(
            metrics.record_batch_prediction,
            batch_size=len(request.samples),
            processing_time=processing_time,
            success=False
        )
        
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.post("/upload/hyperspectral")
async def upload_hyperspectral_image(
    file: UploadFile = File(...),
    server: ModelServer = Depends(get_model_server)
):
    """Upload and process hyperspectral image file."""
    if not file.filename.endswith(('.hdr', '.img', '.tif', '.tiff')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Use .hdr/.img or .tif/.tiff files"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process the file
        processed_data = await server.process_hyperspectral_file(tmp_file_path)
        
        # Cleanup temporary file
        os.unlink(tmp_file_path)
        
        return {
            "message": "File processed successfully",
            "filename": file.filename,
            "data_shape": processed_data["shape"],
            "wavelength_range": processed_data["wavelength_range"],
            "processing_info": processed_data["info"]
        }
        
    except Exception as e:
        # Cleanup on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        logger.error(f"File processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"File processing failed: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info(
    server: ModelServer = Depends(get_model_server)
):
    """Get model information and capabilities."""
    try:
        model_info = await server.get_model_info()
        return model_info
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model information"
        )

@app.post("/model/interpret")
async def interpret_prediction(
    request: PredictionRequest,
    server: ModelServer = Depends(get_model_server)
):
    """Get detailed interpretation for a prediction."""
    try:
        # Convert input data
        image_tensor = torch.tensor(request.image_data, dtype=torch.float32)
        sensor_tensor = torch.tensor(request.sensor_data, dtype=torch.float32)
        
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.unsqueeze(0)
        if sensor_tensor.dim() == 2:
            sensor_tensor = sensor_tensor.unsqueeze(0)
        
        # Get interpretation
        interpretation = await server.get_interpretation(
            image_sequence=image_tensor,
            sensor_sequence=sensor_tensor,
            sequence_length=request.sequence_length
        )
        
        return {
            "interpretation": interpretation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Interpretation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Interpretation failed: {str(e)}"
        )

@app.get("/export/report/{prediction_id}")
async def export_prediction_report(
    prediction_id: str,
    format: str = "pdf"  # pdf, html, json
):
    """Export detailed prediction report."""
    # This would typically retrieve prediction from database
    # For now, return placeholder
    raise HTTPException(
        status_code=501,
        detail="Report export not yet implemented"
    )

@app.websocket("/ws/realtime")
async def websocket_realtime_predictions(websocket):
    """WebSocket endpoint for real-time predictions."""
    await websocket.accept()
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Process prediction (simplified)
            # In practice, you'd convert data and call model server
            result = {
                "prediction": "healthy",
                "confidence": 0.95,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send result back
            await websocket.send_json(result)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )