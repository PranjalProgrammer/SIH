-- Initialize Precision Agriculture Database

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS precision_agriculture;

-- Use the database
\c precision_agriculture;

-- Create tables for agricultural data management

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(100) NOT NULL,
    input_type VARCHAR(50) NOT NULL,
    predicted_class VARCHAR(100) NOT NULL,
    confidence REAL NOT NULL,
    probabilities JSONB,
    processing_time REAL,
    image_path VARCHAR(255),
    sensor_data JSONB,
    weather_data JSONB,
    explanation JSONB,
    user_feedback VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value REAL NOT NULL,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dataset_name VARCHAR(100),
    sample_count INTEGER
);

-- Drift detection results
CREATE TABLE IF NOT EXISTS drift_detection (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detector_type VARCHAR(50) NOT NULL,
    drift_detected BOOLEAN NOT NULL,
    drift_score REAL,
    p_value REAL,
    batch_size INTEGER,
    model_name VARCHAR(100),
    details JSONB
);

-- System alerts
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    source VARCHAR(100),
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP
);

-- Active learning selections
CREATE TABLE IF NOT EXISTS active_learning (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    strategy VARCHAR(50) NOT NULL,
    selected_samples JSONB NOT NULL,
    uncertainty_scores JSONB,
    diversity_scores JSONB,
    batch_size INTEGER,
    total_unlabeled INTEGER
);

-- User feedback
CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES predictions(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feedback_type VARCHAR(20) NOT NULL, -- 'correct', 'incorrect', 'partial'
    correct_label VARCHAR(100),
    confidence_rating INTEGER CHECK (confidence_rating >= 1 AND confidence_rating <= 5),
    comments TEXT,
    user_id VARCHAR(100)
);

-- System metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    unit VARCHAR(20),
    component VARCHAR(50) -- 'api', 'model_server', 'database', etc.
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_class ON predictions(predicted_class);

CREATE INDEX IF NOT EXISTS idx_model_performance_model ON model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_date ON model_performance(evaluation_date);

CREATE INDEX IF NOT EXISTS idx_drift_timestamp ON drift_detection(timestamp);
CREATE INDEX IF NOT EXISTS idx_drift_detected ON drift_detection(drift_detected);

CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved);

CREATE INDEX IF NOT EXISTS idx_feedback_prediction ON user_feedback(prediction_id);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON user_feedback(timestamp);

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_component ON system_metrics(component);

-- Create views for common queries

-- Recent predictions summary
CREATE OR REPLACE VIEW recent_predictions AS
SELECT 
    id,
    timestamp,
    model_name,
    input_type,
    predicted_class,
    confidence,
    processing_time,
    CASE 
        WHEN user_feedback IS NOT NULL THEN user_feedback
        ELSE 'pending'
    END as feedback_status
FROM predictions
WHERE timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;

-- Model performance summary
CREATE OR REPLACE VIEW model_performance_summary AS
SELECT 
    model_name,
    AVG(CASE WHEN metric_name = 'accuracy' THEN metric_value END) as accuracy,
    AVG(CASE WHEN metric_name = 'precision' THEN metric_value END) as precision,
    AVG(CASE WHEN metric_name = 'recall' THEN metric_value END) as recall,
    AVG(CASE WHEN metric_name = 'f1_score' THEN metric_value END) as f1_score,
    MAX(evaluation_date) as last_evaluation
FROM model_performance
GROUP BY model_name;

-- Active alerts
CREATE OR REPLACE VIEW active_alerts AS
SELECT 
    id,
    timestamp,
    alert_type,
    severity,
    message,
    source
FROM alerts
WHERE resolved = FALSE
ORDER BY 
    CASE severity 
        WHEN 'high' THEN 1 
        WHEN 'medium' THEN 2 
        WHEN 'low' THEN 3 
    END,
    timestamp DESC;

-- Insert some sample data for testing
INSERT INTO predictions (model_name, input_type, predicted_class, confidence, processing_time)
VALUES 
    ('CNN Crop Health Model', 'image', 'Healthy', 0.94, 0.23),
    ('LSTM Sensor Model', 'sensor', 'Water Stress', 0.87, 0.15),
    ('Hybrid Model', 'multimodal', 'Nutrient Deficiency', 0.91, 0.45);

INSERT INTO model_performance (model_name, metric_name, metric_value, dataset_name, sample_count)
VALUES 
    ('CNN Crop Health Model', 'accuracy', 0.94, 'test_set', 1000),
    ('CNN Crop Health Model', 'precision', 0.93, 'test_set', 1000),
    ('CNN Crop Health Model', 'recall', 0.95, 'test_set', 1000),
    ('CNN Crop Health Model', 'f1_score', 0.94, 'test_set', 1000);

INSERT INTO alerts (alert_type, severity, message, source)
VALUES 
    ('performance', 'medium', 'Model accuracy dropped below 90% in sector 3', 'model_monitor'),
    ('system', 'low', 'High memory usage detected', 'system_monitor');

-- Grant permissions (adjust as needed for your setup)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO agri_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO agri_user;