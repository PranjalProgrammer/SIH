"""
Sensor Data Processing Module
Handles IoT sensor data processing and integration
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SensorDataProcessor:
    """Processes IoT sensor data for agricultural monitoring"""
    
    def __init__(self):
        self.sensor_types = {
            'soil_moisture': {'unit': '%', 'range': (0, 100)},
            'soil_temperature': {'unit': '°C', 'range': (-10, 50)},
            'air_temperature': {'unit': '°C', 'range': (-20, 60)},
            'humidity': {'unit': '%', 'range': (0, 100)},
            'ph': {'unit': 'pH', 'range': (0, 14)},
            'light_intensity': {'unit': 'lux', 'range': (0, 100000)},
            'rainfall': {'unit': 'mm', 'range': (0, 500)}
        }
    
    def validate_sensor_data(self, data):
        """Validate sensor data against expected ranges"""
        validated_data = {}
        
        for sensor_type, value in data.items():
            if sensor_type in self.sensor_types:
                min_val, max_val = self.sensor_types[sensor_type]['range']
                if min_val <= value <= max_val:
                    validated_data[sensor_type] = value
                else:
                    logger.warning(f"Invalid {sensor_type} value: {value}")
                    validated_data[sensor_type] = np.clip(value, min_val, max_val)
            else:
                validated_data[sensor_type] = value
        
        return validated_data
    
    def aggregate_sensor_data(self, sensor_readings, time_window='1H'):
        """Aggregate sensor data over time windows"""
        df = pd.DataFrame(sensor_readings)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample and aggregate
        aggregated = df.resample(time_window).agg({
            'soil_moisture': 'mean',
            'soil_temperature': 'mean', 
            'air_temperature': 'mean',
            'humidity': 'mean',
            'ph': 'mean',
            'light_intensity': 'mean',
            'rainfall': 'sum'
        })
        
        return aggregated
    
    def detect_anomalies(self, sensor_data):
        """Detect anomalies in sensor readings"""
        anomalies = []
        
        for sensor_type, value in sensor_data.items():
            if sensor_type in self.sensor_types:
                min_val, max_val = self.sensor_types[sensor_type]['range']
                
                # Simple threshold-based anomaly detection
                if value < min_val * 0.1 or value > max_val * 0.9:
                    anomalies.append({
                        'sensor': sensor_type,
                        'value': value,
                        'severity': 'high' if value < min_val * 0.05 or value > max_val * 0.95 else 'medium'
                    })
        
        return anomalies
    
    def generate_alerts(self, anomalies):
        """Generate alerts based on detected anomalies"""
        alerts = []
        
        for anomaly in anomalies:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': 'sensor_anomaly',
                'sensor': anomaly['sensor'],
                'value': anomaly['value'],
                'severity': anomaly['severity'],
                'message': f"Unusual {anomaly['sensor']} reading: {anomaly['value']}"
            }
            alerts.append(alert)
        
        return alerts

def main():
    """Main sensor processing function"""
    processor = SensorDataProcessor()
    logger.info("Sensor data processor initialized")

if __name__ == "__main__":
    main()