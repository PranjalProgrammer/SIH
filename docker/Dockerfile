# Multi-stage Dockerfile for Precision Agriculture Platform

# Base stage with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libspatialite-dev \
    libhdf5-dev \
    libnetcdf-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_DATA=/usr/share/gdal \
    PROJ_LIB=/usr/share/proj

# Development stage
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code
COPY . .

# Expose ports
EXPOSE 8000 3000

# Default command for development
CMD ["python", "-m", "uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash --user-group precision

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (production only)
RUN pip install --upgrade pip && \
    pip install --no-dev -r requirements.txt && \
    pip install gunicorn uvicorn[standard]

# Copy application code
COPY --chown=precision:precision src/ ./src/
COPY --chown=precision:precision configs/ ./configs/
COPY --chown=precision:precision models/ ./models/ 2>/dev/null || true

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models && \
    chown -R precision:precision /app

# Switch to non-root user
USER precision

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300", "src.deployment.api:app"]

# Edge deployment stage (optimized for edge devices)
FROM python:3.9-slim as edge

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy minimal requirements for edge
COPY requirements-edge.txt .
RUN pip install --no-cache-dir -r requirements-edge.txt

# Copy optimized model and minimal code
COPY --from=production /app/src/models/ ./src/models/
COPY --from=production /app/src/deployment/ ./src/deployment/
COPY models/optimized/ ./models/ 2>/dev/null || true

# Edge-specific configuration
ENV MODEL_OPTIMIZATION=true \
    BATCH_SIZE=1 \
    PRECISION=fp16

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.deployment.edge_api:app", "--host", "0.0.0.0", "--port", "8000"]