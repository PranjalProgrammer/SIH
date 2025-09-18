"""
Precision Agriculture Dashboard

Interactive web dashboard for crop health monitoring and analysis.
Built with Streamlit for rapid development and deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
REFRESH_INTERVAL = 30  # seconds

# Page configuration
st.set_page_config(
    page_title="Precision Agriculture Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E7D32;
    }
    .alert-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .success-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class DashboardAPI:
    """API client for dashboard backend communication."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logger.error(f"Metrics request failed: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            response = requests.get(f"{self.base_url}/model/info", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logger.error(f"Model info request failed: {e}")
            return {}
    
    def predict(self, image_data: List, sensor_data: List, **kwargs) -> Dict[str, Any]:
        """Make prediction request."""
        try:
            payload = {
                "image_data": image_data,
                "sensor_data": sensor_data,
                **kwargs
            }
            response = requests.post(f"{self.base_url}/predict", json=payload, timeout=30)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logger.error(f"Prediction request failed: {e}")
            return {"error": str(e)}

# Initialize API client
@st.cache_resource
def get_api_client():
    return DashboardAPI(API_BASE_URL)

# Sample data generators for demonstration
@st.cache_data
def generate_sample_field_data():
    """Generate sample field data for visualization."""
    np.random.seed(42)
    
    # Generate field coordinates
    x = np.linspace(0, 100, 20)
    y = np.linspace(0, 100, 20)
    X, Y = np.meshgrid(x, y)
    
    # Generate NDVI values with spatial patterns
    center_x, center_y = 50, 50
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    ndvi = 0.8 - 0.3 * distance / 70 + np.random.normal(0, 0.05, X.shape)
    ndvi = np.clip(ndvi, 0, 1)
    
    # Create health status based on NDVI
    health_status = np.where(ndvi > 0.7, 'Healthy',
                           np.where(ndvi > 0.4, 'Stressed', 'Diseased'))
    
    # Flatten for DataFrame
    field_data = pd.DataFrame({
        'x': X.flatten(),
        'y': Y.flatten(),
        'ndvi': ndvi.flatten(),
        'health_status': health_status.flatten()
    })
    
    return field_data

@st.cache_data
def generate_time_series_data():
    """Generate sample time series data."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate synthetic sensor data
    np.random.seed(42)
    data = {
        'date': dates,
        'soil_moisture': 45 + 15 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 5, len(dates)),
        'temperature': 20 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 3, len(dates)),
        'humidity': 65 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365 + np.pi) + np.random.normal(0, 8, len(dates)),
        'ndvi': 0.6 + 0.3 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365 + np.pi/4) + np.random.normal(0, 0.05, len(dates))
    }
    
    df = pd.DataFrame(data)
    df['soil_moisture'] = np.clip(df['soil_moisture'], 0, 100)
    df['humidity'] = np.clip(df['humidity'], 0, 100)
    df['ndvi'] = np.clip(df['ndvi'], 0, 1)
    
    return df

def create_field_map(field_data: pd.DataFrame):
    """Create interactive field health map."""
    fig = px.scatter(
        field_data,
        x='x', y='y',
        color='ndvi',
        color_continuous_scale='RdYlGn',
        size_max=15,
        title="Field Health Map (NDVI)",
        labels={'x': 'Field X (m)', 'y': 'Field Y (m)', 'ndvi': 'NDVI'}
    )
    
    fig.update_layout(
        coloraxis_colorbar=dict(title="NDVI"),
        height=500
    )
    
    return fig

def create_time_series_chart(df: pd.DataFrame):
    """Create time series visualization."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Soil Moisture (%)', 'Temperature (°C)', 'Humidity (%)', 'NDVI'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Soil Moisture
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['soil_moisture'], 
                  name='Soil Moisture', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Temperature
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['temperature'], 
                  name='Temperature', line=dict(color='red')),
        row=1, col=2
    )
    
    # Humidity
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['humidity'], 
                  name='Humidity', line=dict(color='green')),
        row=2, col=1
    )
    
    # NDVI
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['ndvi'], 
                  name='NDVI', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Environmental and Vegetation Trends"
    )
    
    return fig

def create_health_distribution_chart(field_data: pd.DataFrame):
    """Create health status distribution chart."""
    health_counts = field_data['health_status'].value_counts()
    
    fig = px.pie(
        values=health_counts.values,
        names=health_counts.index,
        title="Crop Health Distribution",
        color_discrete_map={
            'Healthy': '#4CAF50',
            'Stressed': '#FF9800', 
            'Diseased': '#F44336'
        }
    )
    
    return fig

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<div class="main-header">🌾 Precision Agriculture Platform</div>', 
                unsafe_allow_html=True)
    
    # Initialize API client
    api_client = get_api_client()
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.selectbox("Select Page", [
            "Dashboard Overview",
            "Real-time Monitoring", 
            "Crop Health Analysis",
            "Prediction Tool",
            "System Status",
            "Settings"
        ])
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        
        # Get system health
        health_status = api_client.get_health_status()
        if health_status.get('status') == 'healthy':
            st.success("✅ System Healthy")
        else:
            st.error("❌ System Issues")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        if auto_refresh:
            time.sleep(REFRESH_INTERVAL)
            st.rerun()
    
    # Main content based on selected page
    if page == "Dashboard Overview":
        show_dashboard_overview(api_client)
    elif page == "Real-time Monitoring":
        show_realtime_monitoring(api_client)
    elif page == "Crop Health Analysis":
        show_crop_health_analysis(api_client)
    elif page == "Prediction Tool":
        show_prediction_tool(api_client)
    elif page == "System Status":
        show_system_status(api_client)
    elif page == "Settings":
        show_settings()

def show_dashboard_overview(api_client: DashboardAPI):
    """Show main dashboard overview."""
    st.title("Dashboard Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Fields", "12", "2")
    
    with col2:
        st.metric("Healthy Crops", "78%", "5%")
    
    with col3:
        st.metric("Active Alerts", "3", "-1")
    
    with col4:
        st.metric("Data Points", "1.2M", "15K")
    
    st.markdown("---")
    
    # Main visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Field Health Map")
        field_data = generate_sample_field_data()
        field_map = create_field_map(field_data)
        st.plotly_chart(field_map, use_container_width=True)
    
    with col2:
        st.subheader("Health Distribution")
        health_chart = create_health_distribution_chart(field_data)
        st.plotly_chart(health_chart, use_container_width=True)
    
    # Time series trends
    st.subheader("Environmental Trends")
    time_series_data = generate_time_series_data()
    time_series_chart = create_time_series_chart(time_series_data)
    st.plotly_chart(time_series_chart, use_container_width=True)
    
    # Recent alerts
    st.subheader("Recent Alerts")
    alerts_data = [
        {"Time": "2024-01-15 14:30", "Type": "Disease Risk", "Location": "Field A-3", "Severity": "Medium"},
        {"Time": "2024-01-15 12:15", "Type": "Water Stress", "Location": "Field B-1", "Severity": "High"},
        {"Time": "2024-01-15 09:45", "Type": "Pest Detection", "Location": "Field C-2", "Severity": "Low"}
    ]
    
    alerts_df = pd.DataFrame(alerts_data)
    st.dataframe(alerts_df, use_container_width=True)

def show_realtime_monitoring(api_client: DashboardAPI):
    """Show real-time monitoring page."""
    st.title("Real-time Monitoring")
    
    # Real-time metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Temperature", "24.5°C", "1.2°C")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Soil Moisture", "42%", "-3%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Light Intensity", "850 W/m²", "50 W/m²")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Live sensor data simulation
    st.subheader("Live Sensor Data")
    
    # Create placeholder for real-time chart
    chart_placeholder = st.empty()
    
    # Generate real-time data
    if st.button("Start Real-time Monitoring"):
        for i in range(50):
            # Generate new data point
            current_time = datetime.now() - timedelta(seconds=50-i)
            temperature = 24 + np.random.normal(0, 1)
            humidity = 65 + np.random.normal(0, 5)
            
            # Create real-time chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[current_time],
                y=[temperature],
                mode='markers+lines',
                name='Temperature',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="Real-time Temperature",
                xaxis_title="Time",
                yaxis_title="Temperature (°C)",
                height=400
            )
            
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.1)

def show_crop_health_analysis(api_client: DashboardAPI):
    """Show crop health analysis page."""
    st.title("Crop Health Analysis")
    
    # Analysis controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_field = st.selectbox("Select Field", ["Field A", "Field B", "Field C"])
    
    with col2:
        date_range = st.date_input("Date Range", value=[
            datetime.now() - timedelta(days=30),
            datetime.now()
        ])
    
    with col3:
        analysis_type = st.selectbox("Analysis Type", [
            "NDVI Trends", "Disease Detection", "Stress Analysis"
        ])
    
    # Generate analysis based on selection
    if analysis_type == "NDVI Trends":
        st.subheader("NDVI Trend Analysis")
        
        # Generate sample NDVI data
        dates = pd.date_range(start=date_range[0], end=date_range[1], freq='D')
        ndvi_values = 0.7 + 0.2 * np.sin(np.arange(len(dates)) * 2 * np.pi / 30) + np.random.normal(0, 0.05, len(dates))
        ndvi_values = np.clip(ndvi_values, 0, 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=ndvi_values,
            mode='lines+markers',
            name='NDVI',
            line=dict(color='green', width=2)
        ))
        
        # Add threshold lines
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                     annotation_text="Healthy Threshold")
        fig.add_hline(y=0.4, line_dash="dash", line_color="red", 
                     annotation_text="Stress Threshold")
        
        fig.update_layout(
            title=f"NDVI Trends for {selected_field}",
            xaxis_title="Date",
            yaxis_title="NDVI",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Analysis insights
        st.subheader("Analysis Insights")
        avg_ndvi = np.mean(ndvi_values)
        
        if avg_ndvi > 0.7:
            st.markdown('<div class="success-card">✅ <strong>Healthy Vegetation:</strong> Average NDVI indicates good crop health.</div>', 
                       unsafe_allow_html=True)
        elif avg_ndvi > 0.4:
            st.markdown('<div class="alert-card">⚠️ <strong>Moderate Stress:</strong> NDVI suggests some vegetation stress.</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-card">🚨 <strong>High Stress:</strong> Low NDVI indicates significant vegetation stress.</div>', 
                       unsafe_allow_html=True)

def show_prediction_tool(api_client: DashboardAPI):
    """Show prediction tool page."""
    st.title("Crop Health Prediction Tool")
    
    st.markdown("Upload hyperspectral imagery and sensor data to get AI-powered crop health predictions.")
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hyperspectral Image")
        uploaded_image = st.file_uploader(
            "Upload hyperspectral image", 
            type=['tif', 'tiff', 'hdr'],
            help="Supported formats: TIFF, ENVI"
        )
        
        if uploaded_image:
            st.success(f"Uploaded: {uploaded_image.name}")
    
    with col2:
        st.subheader("Sensor Data")
        uploaded_sensors = st.file_uploader(
            "Upload sensor data",
            type=['csv', 'json'],
            help="CSV or JSON format with sensor readings"
        )
        
        if uploaded_sensors:
            st.success(f"Uploaded: {uploaded_sensors.name}")
    
    # Manual data input option
    st.subheader("Or Enter Data Manually")
    
    with st.expander("Manual Data Entry"):
        # Sensor data input
        col1, col2, col3 = st.columns(3)
        
        with col1:
            soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 45)
            temperature = st.slider("Temperature (°C)", -10, 50, 25)
        
        with col2:
            humidity = st.slider("Humidity (%)", 0, 100, 65)
            light_intensity = st.slider("Light Intensity (W/m²)", 0, 2000, 800)
        
        with col3:
            wind_speed = st.slider("Wind Speed (m/s)", 0, 20, 5)
            rainfall = st.slider("Rainfall (mm)", 0, 50, 2)
        
        # Generate sample data for prediction
        if st.button("Generate Sample Data & Predict"):
            with st.spinner("Processing prediction..."):
                # Create sample data
                sample_image_data = np.random.rand(12, 32, 32, 200).tolist()
                sample_sensor_data = [[
                    soil_moisture, temperature, humidity, light_intensity, 
                    wind_speed, rainfall, 0.8, 25, 1013, 7.0, 100, 50, 200, 0.5, 0.3
                ] for _ in range(12)]
                
                # Make prediction
                prediction_result = api_client.predict(
                    image_data=sample_image_data,
                    sensor_data=sample_sensor_data,
                    include_interpretation=True,
                    include_ood=True
                )
                
                if prediction_result and 'prediction' in prediction_result:
                    show_prediction_results(prediction_result)
                else:
                    st.error("Prediction failed. Please check your data and try again.")

def show_prediction_results(prediction_result: Dict[str, Any]):
    """Display prediction results."""
    prediction = prediction_result['prediction']
    
    st.subheader("Prediction Results")
    
    # Main prediction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Predicted Health Status",
            prediction['predicted_class_name'].title(),
            f"{prediction['confidence']:.2%} confidence"
        )
    
    with col2:
        processing_time = prediction_result.get('processing_time', 0)
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    with col3:
        model_version = prediction_result.get('model_version', 'Unknown')
        st.metric("Model Version", model_version)
    
    # Class probabilities
    st.subheader("Class Probabilities")
    class_names = prediction['class_names']
    probabilities = prediction['class_probabilities']
    
    prob_df = pd.DataFrame({
        'Class': class_names,
        'Probability': probabilities
    })
    
    fig = px.bar(prob_df, x='Class', y='Probability', 
                title="Prediction Confidence by Class",
                color='Probability', color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation results
    if 'interpretation' in prediction_result and prediction_result['interpretation']:
        st.subheader("Model Interpretation")
        interpretation = prediction_result['interpretation']
        
        if 'explanation_summary' in interpretation:
            summary = interpretation['explanation_summary']
            for key, explanation in summary.items():
                st.write(f"**{key.title()}:** {explanation}")

def show_system_status(api_client: DashboardAPI):
    """Show system status page."""
    st.title("System Status")
    
    # System health
    health_status = api_client.get_health_status()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Health")
        if health_status.get('status') == 'healthy':
            st.success("🟢 System is healthy")
        else:
            st.error("🔴 System has issues")
        
        if health_status:
            st.json(health_status)
    
    with col2:
        st.subheader("Performance Metrics")
        metrics = api_client.get_metrics()
        
        if metrics:
            st.metric("Total Predictions", metrics.get('predictions_count', 0))
            st.metric("Average Response Time", f"{metrics.get('average_processing_time', 0):.3f}s")
            st.metric("Error Rate", f"{metrics.get('error_rate', 0):.2%}")
        else:
            st.info("Metrics not available")
    
    # Model information
    st.subheader("Model Information")
    model_info = api_client.get_model_info()
    
    if model_info:
        with st.expander("Model Details"):
            st.json(model_info)
    else:
        st.info("Model information not available")

def show_settings():
    """Show settings page."""
    st.title("Settings")
    
    st.subheader("Dashboard Settings")
    
    # Display settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Theme", ["Light", "Dark", "Auto"])
        st.slider("Refresh Interval (seconds)", 10, 300, 30)
        st.checkbox("Show Notifications", value=True)
    
    with col2:
        st.selectbox("Default Field View", ["All Fields", "Field A", "Field B", "Field C"])
        st.selectbox("Chart Style", ["Modern", "Classic", "Minimal"])
        st.checkbox("Auto-save Reports", value=False)
    
    st.subheader("API Settings")
    
    api_url = st.text_input("API Base URL", value=API_BASE_URL)
    api_timeout = st.slider("API Timeout (seconds)", 5, 60, 30)
    
    if st.button("Test API Connection"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API connection successful")
            else:
                st.error(f"❌ API connection failed: {response.status_code}")
        except Exception as e:
            st.error(f"❌ API connection failed: {str(e)}")
    
    st.subheader("Export Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox("Default Export Format", ["PDF", "CSV", "JSON", "Excel"])
        include_charts = st.checkbox("Include Charts in Reports", value=True)
    
    with col2:
        report_frequency = st.selectbox("Automated Report Frequency", ["Daily", "Weekly", "Monthly", "Never"])
        email_reports = st.text_input("Email for Reports", placeholder="user@example.com")
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

if __name__ == "__main__":
    main()