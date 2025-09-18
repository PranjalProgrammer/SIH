"""
Sensor Data Processing Module

Handles IoT sensor data cleaning, normalization, interpolation,
and temporal synchronization with hyperspectral imagery.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import interpolate
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorDataProcessor:
    """Processes IoT sensor data for precision agriculture applications."""
    
    def __init__(self, data_dir: str = "/workspace/data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Define sensor data schema and valid ranges
        self.sensor_schema = {
            'soil_moisture': {'min': 0, 'max': 100, 'unit': '%'},
            'soil_temperature': {'min': -20, 'max': 60, 'unit': '°C'},
            'air_temperature': {'min': -40, 'max': 60, 'unit': '°C'},
            'humidity': {'min': 0, 'max': 100, 'unit': '%'},
            'leaf_wetness': {'min': 0, 'max': 24, 'unit': 'hours'},
            'light_intensity': {'min': 0, 'max': 100000, 'unit': 'lux'},
            'wind_speed': {'min': 0, 'max': 50, 'unit': 'm/s'},
            'rainfall': {'min': 0, 'max': 200, 'unit': 'mm'},
            'ph_level': {'min': 3, 'max': 10, 'unit': 'pH'},
            'nitrogen': {'min': 0, 'max': 1000, 'unit': 'ppm'},
            'phosphorus': {'min': 0, 'max': 500, 'unit': 'ppm'},
            'potassium': {'min': 0, 'max': 2000, 'unit': 'ppm'}
        }
    
    def load_sensor_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load sensor data from CSV file.
        
        Args:
            file_path: Path to sensor data CSV file
            
        Returns:
            DataFrame with sensor data
        """
        logger.info(f"Loading sensor data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Ensure timestamp column exists and is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                logger.warning("No timestamp column found, creating sequential timestamps")
                df['timestamp'] = pd.date_range(
                    start='2023-01-01', 
                    periods=len(df), 
                    freq='H'
                )
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} sensor records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load sensor data: {str(e)}")
            raise
    
    def validate_sensor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate sensor data against expected ranges and flag outliers.
        
        Args:
            df: Input sensor DataFrame
            
        Returns:
            DataFrame with validation flags
        """
        logger.info("Validating sensor data...")
        
        df_validated = df.copy()
        
        # Add validation columns
        for sensor, limits in self.sensor_schema.items():
            if sensor in df.columns:
                col_name = f"{sensor}_valid"
                df_validated[col_name] = (
                    (df[sensor] >= limits['min']) & 
                    (df[sensor] <= limits['max'])
                )
                
                # Count invalid values
                invalid_count = (~df_validated[col_name]).sum()
                if invalid_count > 0:
                    logger.warning(f"{sensor}: {invalid_count} invalid values found")
        
        logger.info("Sensor data validation completed")
        return df_validated
    
    def clean_sensor_data(self, 
                         df: pd.DataFrame, 
                         method: str = 'interpolation') -> pd.DataFrame:
        """
        Clean sensor data by handling missing values and outliers.
        
        Args:
            df: Input sensor DataFrame
            method: Cleaning method ('interpolation', 'knn', 'forward_fill', 'mean')
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning sensor data using {method} method...")
        
        df_cleaned = df.copy()
        
        # Get sensor columns (exclude timestamp and validation columns)
        sensor_cols = [col for col in df.columns 
                      if col in self.sensor_schema and col in df.columns]
        
        # Handle missing values
        if method == 'interpolation':
            for col in sensor_cols:
                df_cleaned[col] = df_cleaned[col].interpolate(method='linear')
        
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            df_cleaned[sensor_cols] = imputer.fit_transform(df_cleaned[sensor_cols])
        
        elif method == 'forward_fill':
            df_cleaned[sensor_cols] = df_cleaned[sensor_cols].fillna(method='ffill')
        
        elif method == 'mean':
            imputer = SimpleImputer(strategy='mean')
            df_cleaned[sensor_cols] = imputer.fit_transform(df_cleaned[sensor_cols])
        
        # Remove extreme outliers using IQR method
        df_cleaned = self._remove_outliers_iqr(df_cleaned, sensor_cols)
        
        logger.info("Sensor data cleaning completed")
        return df_cleaned
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Remove outliers using Interquartile Range method."""
        df_no_outliers = df.copy()
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Replace outliers with NaN and then interpolate
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                df_no_outliers.loc[outlier_mask, col] = np.nan
                df_no_outliers[col] = df_no_outliers[col].interpolate()
                
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    logger.info(f"{col}: Removed {outlier_count} outliers")
        
        return df_no_outliers
    
    def normalize_sensor_data(self, 
                            df: pd.DataFrame, 
                            method: str = 'minmax') -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize sensor data for ML model training.
        
        Args:
            df: Input sensor DataFrame
            method: Normalization method ('minmax', 'standard', 'robust')
            
        Returns:
            Normalized DataFrame and scaler objects
        """
        logger.info(f"Normalizing sensor data using {method} method...")
        
        df_normalized = df.copy()
        scalers = {}
        
        # Get sensor columns
        sensor_cols = [col for col in df.columns 
                      if col in self.sensor_schema and col in df.columns]
        
        # Apply normalization
        for col in sensor_cols:
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            df_normalized[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler
        
        logger.info("Sensor data normalization completed")
        return df_normalized, scalers
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamp data.
        
        Args:
            df: Input sensor DataFrame with timestamp column
            
        Returns:
            DataFrame with additional temporal features
        """
        logger.info("Extracting temporal features...")
        
        df_temporal = df.copy()
        
        if 'timestamp' in df.columns:
            # Extract time components
            df_temporal['hour'] = df_temporal['timestamp'].dt.hour
            df_temporal['day'] = df_temporal['timestamp'].dt.day
            df_temporal['month'] = df_temporal['timestamp'].dt.month
            df_temporal['year'] = df_temporal['timestamp'].dt.year
            df_temporal['day_of_week'] = df_temporal['timestamp'].dt.dayofweek
            df_temporal['day_of_year'] = df_temporal['timestamp'].dt.dayofyear
            
            # Create cyclical features for better ML representation
            df_temporal['hour_sin'] = np.sin(2 * np.pi * df_temporal['hour'] / 24)
            df_temporal['hour_cos'] = np.cos(2 * np.pi * df_temporal['hour'] / 24)
            df_temporal['month_sin'] = np.sin(2 * np.pi * df_temporal['month'] / 12)
            df_temporal['month_cos'] = np.cos(2 * np.pi * df_temporal['month'] / 12)
            df_temporal['day_sin'] = np.sin(2 * np.pi * df_temporal['day_of_week'] / 7)
            df_temporal['day_cos'] = np.cos(2 * np.pi * df_temporal['day_of_week'] / 7)
            
            logger.info("Temporal features extracted successfully")
        else:
            logger.warning("No timestamp column found, skipping temporal features")
        
        return df_temporal
    
    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived agricultural features from sensor data.
        
        Args:
            df: Input sensor DataFrame
            
        Returns:
            DataFrame with derived features
        """
        logger.info("Calculating derived agricultural features...")
        
        df_derived = df.copy()
        
        # Vapor Pressure Deficit (VPD)
        if 'air_temperature' in df.columns and 'humidity' in df.columns:
            # Saturation vapor pressure (kPa)
            svp = 0.6108 * np.exp(17.27 * df['air_temperature'] / (df['air_temperature'] + 237.3))
            # Actual vapor pressure
            avp = svp * df['humidity'] / 100
            # VPD
            df_derived['vpd'] = svp - avp
        
        # Growing Degree Days (GDD) - base temperature 10°C
        if 'air_temperature' in df.columns:
            df_derived['gdd'] = np.maximum(df['air_temperature'] - 10, 0)
        
        # Soil Water Deficit
        if 'soil_moisture' in df.columns:
            field_capacity = 40  # Assumed field capacity
            df_derived['soil_water_deficit'] = np.maximum(field_capacity - df['soil_moisture'], 0)
        
        # Heat Index
        if 'air_temperature' in df.columns and 'humidity' in df.columns:
            temp_f = df['air_temperature'] * 9/5 + 32  # Convert to Fahrenheit
            rh = df['humidity']
            
            # Simplified heat index calculation
            hi = (-42.379 + 2.04901523*temp_f + 10.14333127*rh - 
                  0.22475541*temp_f*rh - 0.00683783*temp_f**2 - 
                  0.05481717*rh**2 + 0.00122874*temp_f**2*rh + 
                  0.00085282*temp_f*rh**2 - 0.00000199*temp_f**2*rh**2)
            
            df_derived['heat_index'] = (hi - 32) * 5/9  # Convert back to Celsius
        
        # Wind Chill (for cold conditions)
        if 'air_temperature' in df.columns and 'wind_speed' in df.columns:
            temp_f = df['air_temperature'] * 9/5 + 32
            wind_mph = df['wind_speed'] * 2.237  # m/s to mph
            
            # Wind chill for temperatures below 10°C and wind > 4.8 km/h
            mask = (df['air_temperature'] < 10) & (df['wind_speed'] > 1.34)
            
            wc = (35.74 + 0.6215*temp_f - 35.75*(wind_mph**0.16) + 
                  0.4275*temp_f*(wind_mph**0.16))
            
            df_derived['wind_chill'] = np.where(mask, (wc - 32) * 5/9, df['air_temperature'])
        
        logger.info("Derived agricultural features calculated")
        return df_derived
    
    def synchronize_with_imagery(self, 
                               sensor_df: pd.DataFrame,
                               image_timestamps: List[datetime],
                               tolerance: timedelta = timedelta(hours=1)) -> pd.DataFrame:
        """
        Synchronize sensor data with hyperspectral imagery timestamps.
        
        Args:
            sensor_df: Sensor data DataFrame
            image_timestamps: List of image acquisition timestamps
            tolerance: Maximum time difference for matching
            
        Returns:
            Synchronized sensor data
        """
        logger.info("Synchronizing sensor data with imagery timestamps...")
        
        synchronized_data = []
        
        for img_timestamp in image_timestamps:
            # Find closest sensor reading within tolerance
            time_diff = abs(sensor_df['timestamp'] - img_timestamp)
            closest_idx = time_diff.idxmin()
            
            if time_diff.loc[closest_idx] <= tolerance:
                sensor_record = sensor_df.loc[closest_idx].copy()
                sensor_record['image_timestamp'] = img_timestamp
                sensor_record['time_diff_minutes'] = time_diff.loc[closest_idx].total_seconds() / 60
                synchronized_data.append(sensor_record)
            else:
                logger.warning(f"No sensor data within tolerance for image timestamp: {img_timestamp}")
        
        if synchronized_data:
            sync_df = pd.DataFrame(synchronized_data)
            logger.info(f"Synchronized {len(sync_df)} sensor-image pairs")
            return sync_df
        else:
            logger.warning("No synchronized data found")
            return pd.DataFrame()
    
    def create_time_series_windows(self, 
                                 df: pd.DataFrame,
                                 window_size: int = 24,
                                 step_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time series windows for sequence modeling.
        
        Args:
            df: Input sensor DataFrame
            window_size: Size of each time window (hours)
            step_size: Step size between windows
            
        Returns:
            Feature windows and timestamps
        """
        logger.info(f"Creating time series windows (size={window_size}, step={step_size})...")
        
        # Get sensor columns
        sensor_cols = [col for col in df.columns 
                      if col in self.sensor_schema and col in df.columns]
        
        # Add derived features
        feature_cols = sensor_cols + [col for col in df.columns 
                                    if any(x in col for x in ['_sin', '_cos', 'gdd', 'vpd', 'deficit'])]
        
        # Create windows
        windows = []
        timestamps = []
        
        for i in range(0, len(df) - window_size + 1, step_size):
            window_data = df.iloc[i:i + window_size][feature_cols].values
            window_timestamp = df.iloc[i + window_size - 1]['timestamp']
            
            windows.append(window_data)
            timestamps.append(window_timestamp)
        
        windows_array = np.array(windows)
        
        logger.info(f"Created {len(windows)} time series windows of shape {windows_array.shape}")
        return windows_array, timestamps
    
    def process_sensor_pipeline(self, 
                              file_path: Union[str, Path],
                              clean_method: str = 'interpolation',
                              normalize_method: str = 'minmax') -> Tuple[pd.DataFrame, Dict]:
        """
        Complete sensor data processing pipeline.
        
        Args:
            file_path: Path to sensor data file
            clean_method: Data cleaning method
            normalize_method: Normalization method
            
        Returns:
            Processed DataFrame and metadata
        """
        logger.info("Starting complete sensor data processing pipeline...")
        
        # Load data
        df = self.load_sensor_data(file_path)
        
        # Validate data
        df = self.validate_sensor_data(df)
        
        # Clean data
        df = self.clean_sensor_data(df, method=clean_method)
        
        # Extract temporal features
        df = self.extract_temporal_features(df)
        
        # Calculate derived features
        df = self.calculate_derived_features(df)
        
        # Normalize data
        df_normalized, scalers = self.normalize_sensor_data(df, method=normalize_method)
        
        # Prepare metadata
        metadata = {
            'original_records': len(df),
            'processed_records': len(df_normalized),
            'cleaning_method': clean_method,
            'normalization_method': normalize_method,
            'scalers': scalers,
            'feature_columns': list(df_normalized.columns)
        }
        
        logger.info("Sensor data processing pipeline completed")
        return df_normalized, metadata


def main():
    """Main function to demonstrate sensor data processing."""
    processor = SensorDataProcessor()
    
    # Check if sample data exists
    sample_file = processor.raw_dir / 'sample_iot_sensor_data.csv'
    
    if not sample_file.exists():
        logger.info("Creating sample IoT sensor data...")
        # Create sample data similar to download_datasets.py
        from datetime import datetime, timedelta
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        start_date = datetime.now() - timedelta(days=30)
        dates = [start_date + timedelta(hours=x) for x in range(720)]
        
        data = {
            'timestamp': dates,
            'soil_moisture': np.random.normal(45, 10, len(dates)).clip(0, 100),
            'soil_temperature': np.random.normal(22, 5, len(dates)).clip(-10, 50),
            'air_temperature': np.random.normal(25, 8, len(dates)).clip(-20, 50),
            'humidity': np.random.normal(65, 15, len(dates)).clip(0, 100),
            'leaf_wetness': np.random.exponential(2, len(dates)).clip(0, 24),
            'light_intensity': np.random.gamma(2, 2, len(dates)) * 1000,
            'wind_speed': np.random.exponential(1.5, len(dates)).clip(0, 20),
            'rainfall': np.random.exponential(0.5, len(dates)).clip(0, 50)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(sample_file, index=False)
    
    # Process sensor data
    processed_df, metadata = processor.process_sensor_pipeline(sample_file)
    
    # Save processed data
    output_path = processor.processed_dir / 'processed_sensor_data.csv'
    processed_df.to_csv(output_path, index=False)
    
    # Create time series windows
    windows, timestamps = processor.create_time_series_windows(processed_df, window_size=24)
    
    # Save windows
    np.save(processor.processed_dir / 'sensor_windows.npy', windows)
    
    logger.info(f"Processed sensor data saved to: {output_path}")
    logger.info(f"Time series windows saved: {windows.shape}")
    logger.info(f"Processing metadata: {metadata}")


if __name__ == "__main__":
    main()