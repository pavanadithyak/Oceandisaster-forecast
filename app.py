import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import os
from datetime import datetime, timedelta
import time

# Import custom modules
from data.api_client import MarineDataClient
from models.bayesian_lstm import BayesianLSTM
from models.arima_baseline import ARIMABaseline
from utils.preprocessing import TimeSeriesPreprocessor
from utils.metrics import ModelMetrics
from utils.risk_assessment import RiskAssessment

# Page configuration
st.set_page_config(
    page_title="Ocean Forecasting Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
.risk-alert {
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.risk-high {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
}
.risk-medium {
    background-color: #fff3e0;
    border-left: 4px solid #ff9800;
}
.risk-low {
    background-color: #e8f5e8;
    border-left: 4px solid #4caf50;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# Title and description
st.title("🌊 Ocean Forecasting Dashboard")
st.markdown("Real-time marine data analysis with Bayesian LSTM modeling and uncertainty quantification")

# Sidebar controls
st.sidebar.header("Configuration")

# Location selection
locations = {
    "New York Harbor": {"lat": 40.7128, "lon": -74.0060},
    "Miami Beach": {"lat": 25.7617, "lon": -80.1918},
    "San Francisco Bay": {"lat": 37.7749, "lon": -122.4194},
    "Boston Harbor": {"lat": 42.3601, "lon": -71.0589},
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437}
}

selected_location = st.sidebar.selectbox(
    "Select Location",
    options=list(locations.keys()),
    index=0
)

# Parameter selection
parameters = {
    "Sea Level": "sea_level",
    "Wind Speed": "wind_speed_10m",
    "Atmospheric Pressure": "pressure_msl",
    "Temperature": "temperature_2m"
}

selected_parameter = st.sidebar.selectbox(
    "Select Parameter",
    options=list(parameters.keys()),
    index=0
)

# Forecast horizon
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (hours)",
    min_value=6,
    max_value=72,
    value=24,
    step=6
)

# Model configuration
st.sidebar.subheader("Model Configuration")
sequence_length = st.sidebar.slider("Sequence Length", 12, 72, 24)
hidden_size = st.sidebar.slider("LSTM Hidden Size", 32, 128, 64)
num_layers = st.sidebar.slider("Number of LSTM Layers", 1, 4, 2)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.2)
mc_samples = st.sidebar.slider("Monte Carlo Samples", 50, 200, 100)

# Auto-refresh option
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)

# Data fetching and caching
@st.cache_data(ttl=900)  # Cache for 15 minutes
def fetch_marine_data(lat, lon, days=7):
    """Fetch marine data with caching"""
    client = MarineDataClient()
    try:
        data = client.get_marine_data(lat, lon, days)
        return data, None
    except Exception as e:
        return None, str(e)

# Initialize components
client = MarineDataClient()
preprocessor = TimeSeriesPreprocessor()
metrics_calculator = ModelMetrics()
risk_assessor = RiskAssessment()

# Main dashboard
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.subheader("📍 Location Information")
    lat = locations[selected_location]["lat"]
    lon = locations[selected_location]["lon"]
    st.write(f"**{selected_location}**")
    st.write(f"Latitude: {lat:.4f}, Longitude: {lon:.4f}")

with col2:
    st.subheader("🔄 Data Status")
    if st.button("Refresh Data") or auto_refresh:
        st.session_state.last_update = datetime.now()
        # Clear cache to force refresh
        st.cache_data.clear()

with col3:
    st.subheader("⏰ Last Updated")
    if st.session_state.last_update:
        st.write(st.session_state.last_update.strftime("%H:%M:%S"))
    else:
        st.write("Never")

# Fetch data
with st.spinner("Fetching marine data..."):
    data, error = fetch_marine_data(lat, lon)

if error:
    st.error(f"Error fetching data: {error}")
    st.stop()

if data is None or data.empty:
    st.error("No data available for the selected location")
    st.stop()

# Display current conditions
st.subheader("🌡️ Current Conditions")
current_data = data.iloc[-1]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Sea Level (m)",
        f"{current_data.get('sea_level', 0):.2f}",
        delta=f"{current_data.get('sea_level', 0) - data.iloc[-2].get('sea_level', 0):.2f}" if len(data) > 1 else None
    )

with col2:
    st.metric(
        "Wind Speed (m/s)",
        f"{current_data.get('wind_speed_10m', 0):.1f}",
        delta=f"{current_data.get('wind_speed_10m', 0) - data.iloc[-2].get('wind_speed_10m', 0):.1f}" if len(data) > 1 else None
    )

with col3:
    st.metric(
        "Pressure (hPa)",
        f"{current_data.get('pressure_msl', 0):.1f}",
        delta=f"{current_data.get('pressure_msl', 0) - data.iloc[-2].get('pressure_msl', 0):.1f}" if len(data) > 1 else None
    )

with col4:
    st.metric(
        "Temperature (°C)",
        f"{current_data.get('temperature_2m', 0):.1f}",
        delta=f"{current_data.get('temperature_2m', 0) - data.iloc[-2].get('temperature_2m', 0):.1f}" if len(data) > 1 else None
    )

# Preprocessing
with st.spinner("Preprocessing data..."):
    processed_data = preprocessor.preprocess(data)
    sequences, targets = preprocessor.create_sequences(
        processed_data[parameters[selected_parameter]].values,
        sequence_length
    )

if len(sequences) < 10:
    st.error("Insufficient data for modeling. Need at least 10 sequences.")
    st.stop()

# Model training and prediction
st.subheader("🤖 Model Training and Prediction")

col1, col2 = st.columns(2)

with col1:
    st.write("**Bayesian LSTM Configuration**")
    st.write(f"- Sequence Length: {sequence_length}")
    st.write(f"- Hidden Size: {hidden_size}")
    st.write(f"- Number of Layers: {num_layers}")
    st.write(f"- Dropout Rate: {dropout_rate}")
    st.write(f"- Monte Carlo Samples: {mc_samples}")

with col2:
    st.write("**Training Progress**")
    progress_bar = st.progress(0)
    status_text = st.empty()

# Train models
try:
    # Bayesian LSTM
    status_text.text("Training Bayesian LSTM...")
    lstm_model = BayesianLSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )
    
    train_loader = lstm_model.prepare_data(sequences, targets)
    lstm_model.train_model(train_loader, epochs=50, progress_callback=lambda p: progress_bar.progress(p))
    
    # Generate predictions
    status_text.text("Generating predictions...")
    predictions, uncertainties = lstm_model.predict_with_uncertainty(
        sequences[-1:], forecast_horizon, mc_samples
    )
    
    # ARIMA baseline
    status_text.text("Training ARIMA baseline...")
    arima_model = ARIMABaseline()
    arima_predictions = arima_model.fit_predict(
        processed_data[parameters[selected_parameter]].values,
        forecast_horizon
    )
    
    progress_bar.progress(1.0)
    status_text.text("Training completed!")
    
except Exception as e:
    st.error(f"Error during model training: {str(e)}")
    st.stop()

# Calculate metrics
lstm_metrics = metrics_calculator.calculate_metrics(
    targets[-len(predictions):] if len(targets) >= len(predictions) else targets,
    predictions[:len(targets)] if len(targets) < len(predictions) else predictions
)

arima_metrics = metrics_calculator.calculate_metrics(
    targets[-len(arima_predictions):] if len(targets) >= len(arima_predictions) else targets,
    arima_predictions[:len(targets)] if len(targets) < len(arima_predictions) else arima_predictions
)

# Display metrics
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.write("**Bayesian LSTM**")
    st.metric("R² Score", f"{lstm_metrics['r2']:.4f}")
    st.metric("RMSE", f"{lstm_metrics['rmse']:.4f}")
    st.metric("MAE", f"{lstm_metrics['mae']:.4f}")

with col2:
    st.write("**ARIMA Baseline**")
    st.metric("R² Score", f"{arima_metrics['r2']:.4f}")
    st.metric("RMSE", f"{arima_metrics['rmse']:.4f}")
    st.metric("MAE", f"{arima_metrics['mae']:.4f}")

# Risk assessment
st.subheader("⚠️ Risk Assessment")
risk_level, risk_message = risk_assessor.assess_risk(
    current_data.get(parameters[selected_parameter], 0),
    predictions,
    uncertainties,
    selected_parameter
)

if risk_level == "High":
    st.markdown(f'<div class="risk-alert risk-high"><strong>🚨 HIGH RISK:</strong> {risk_message}</div>', unsafe_allow_html=True)
elif risk_level == "Medium":
    st.markdown(f'<div class="risk-alert risk-medium"><strong>⚠️ MEDIUM RISK:</strong> {risk_message}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="risk-alert risk-low"><strong>✅ LOW RISK:</strong> {risk_message}</div>', unsafe_allow_html=True)

# Visualization
st.subheader("📈 Forecasting Visualization")

# Create time indices
historical_time = pd.date_range(
    end=datetime.now(),
    periods=len(processed_data),
    freq='h'
)

forecast_time = pd.date_range(
    start=historical_time[-1] + timedelta(hours=1),
    periods=forecast_horizon,
    freq='h'
)

# Bayesian LSTM Plot
st.subheader("🧠 Bayesian LSTM Forecast with Uncertainty")
lstm_fig = go.Figure()

# Historical data
lstm_fig.add_trace(
    go.Scatter(
        x=historical_time,
        y=processed_data[parameters[selected_parameter]],
        mode='lines',
        name='Historical Data',
        line=dict(color='blue', width=2)
    )
)

# Predictions with uncertainty bands
upper_bound = predictions + 1.96 * uncertainties  # 95% confidence interval
lower_bound = predictions - 1.96 * uncertainties

lstm_fig.add_trace(
    go.Scatter(
        x=forecast_time,
        y=upper_bound,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hovertemplate='Upper Bound: %{y:.3f}<extra></extra>'
    )
)

lstm_fig.add_trace(
    go.Scatter(
        x=forecast_time,
        y=lower_bound,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(width=0),
        name='95% Confidence Interval',
        hovertemplate='Lower Bound: %{y:.3f}<extra></extra>'
    )
)

lstm_fig.add_trace(
    go.Scatter(
        x=forecast_time,
        y=predictions,
        mode='lines+markers',
        name='LSTM Prediction',
        line=dict(color='red', width=3),
        marker=dict(size=6, color='red')
    )
)

lstm_fig.update_layout(
    title=f"Bayesian LSTM Forecast - {selected_parameter} at {selected_location}",
    xaxis_title="Time",
    yaxis_title=f"{selected_parameter}",
    height=500,
    showlegend=True,
    hovermode='x unified'
)

st.plotly_chart(lstm_fig, use_container_width=True)

# ARIMA Model Plot
st.subheader("📊 ARIMA Baseline Forecast")
arima_fig = go.Figure()

# Historical data
arima_fig.add_trace(
    go.Scatter(
        x=historical_time,
        y=processed_data[parameters[selected_parameter]],
        mode='lines',
        name='Historical Data',
        line=dict(color='blue', width=2)
    )
)

# ARIMA predictions
arima_fig.add_trace(
    go.Scatter(
        x=forecast_time,
        y=arima_predictions,
        mode='lines+markers',
        name='ARIMA Prediction',
        line=dict(color='green', width=3, dash='dash'),
        marker=dict(size=6, color='green', symbol='diamond')
    )
)

arima_fig.update_layout(
    title=f"ARIMA Baseline Forecast - {selected_parameter} at {selected_location}",
    xaxis_title="Time",
    yaxis_title=f"{selected_parameter}",
    height=500,
    showlegend=True,
    hovermode='x unified'
)

st.plotly_chart(arima_fig, use_container_width=True)

# Model Comparison Plot
st.subheader("⚖️ Model Performance Comparison")
comparison_fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=['R² Score Comparison', 'RMSE Comparison'],
    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
)

models = ['Bayesian LSTM', 'ARIMA']
r2_scores = [lstm_metrics['r2'], arima_metrics['r2']]
rmse_scores = [lstm_metrics['rmse'], arima_metrics['rmse']]

# R² Score comparison
comparison_fig.add_trace(
    go.Bar(
        x=models,
        y=r2_scores,
        name='R² Score',
        marker_color=['#ff6b6b', '#4ecdc4'],
        text=[f'{score:.4f}' for score in r2_scores],
        textposition='auto'
    ),
    row=1, col=1
)

# RMSE comparison
comparison_fig.add_trace(
    go.Bar(
        x=models,
        y=rmse_scores,
        name='RMSE',
        marker_color=['#ff9ff3', '#54a0ff'],
        text=[f'{score:.4f}' for score in rmse_scores],
        textposition='auto'
    ),
    row=1, col=2
)

comparison_fig.update_layout(
    title_text="Model Performance Metrics Comparison",
    height=400,
    showlegend=False
)

comparison_fig.update_xaxes(title_text="Model", row=1, col=1)
comparison_fig.update_yaxes(title_text="R² Score", row=1, col=1)
comparison_fig.update_xaxes(title_text="Model", row=1, col=2)
comparison_fig.update_yaxes(title_text="RMSE", row=1, col=2)

st.plotly_chart(comparison_fig, use_container_width=True)

# Combined Overlay Plot (Optional)
with st.expander("📈 View Combined Overlay Comparison"):
    combined_fig = go.Figure()
    
    # Historical data
    combined_fig.add_trace(
        go.Scatter(
            x=historical_time,
            y=processed_data[parameters[selected_parameter]],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue', width=2)
        )
    )
    
    # LSTM predictions
    combined_fig.add_trace(
        go.Scatter(
            x=forecast_time,
            y=predictions,
            mode='lines+markers',
            name='Bayesian LSTM',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        )
    )
    
    # ARIMA predictions
    combined_fig.add_trace(
        go.Scatter(
            x=forecast_time,
            y=arima_predictions,
            mode='lines+markers',
            name='ARIMA Baseline',
            line=dict(color='green', width=2, dash='dash'),
            marker=dict(size=4, symbol='diamond')
        )
    )
    
    # Uncertainty band for LSTM
    combined_fig.add_trace(
        go.Scatter(
            x=forecast_time,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        )
    )
    
    combined_fig.add_trace(
        go.Scatter(
            x=forecast_time,
            y=lower_bound,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(width=0),
            name='LSTM 95% CI'
        )
    )
    
    combined_fig.update_layout(
        title=f"Combined Model Comparison - {selected_parameter} at {selected_location}",
        xaxis_title="Time",
        yaxis_title=f"{selected_parameter}",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(combined_fig, use_container_width=True)

# Additional visualizations
col1, col2 = st.columns(2)

with col1:
    # Uncertainty visualization
    uncertainty_fig = go.Figure()
    uncertainty_fig.add_trace(
        go.Scatter(
            x=forecast_time,
            y=uncertainties,
            mode='lines+markers',
            name='Prediction Uncertainty',
            line=dict(color='orange', width=2)
        )
    )
    uncertainty_fig.update_layout(
        title="Prediction Uncertainty Over Time",
        xaxis_title="Time",
        yaxis_title="Uncertainty (±)",
        height=400
    )
    st.plotly_chart(uncertainty_fig, use_container_width=True)

with col2:
    # Parameter correlation heatmap
    correlation_matrix = processed_data[list(parameters.values())].corr()
    heatmap_fig = px.imshow(
        correlation_matrix,
        labels=dict(x="Parameters", y="Parameters", color="Correlation"),
        x=list(parameters.keys()),
        y=list(parameters.keys()),
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    heatmap_fig.update_layout(
        title="Parameter Correlation Matrix",
        height=400
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

# Data table
st.subheader("📋 Recent Data")
st.dataframe(
    data.tail(10)[list(parameters.values()) + ['time']].round(3),
    use_container_width=True
)

# Auto-refresh functionality
if auto_refresh:
    time.sleep(30)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("🌊 Ocean Forecasting Dashboard | Built with Streamlit, PyTorch, and Open-Meteo Marine API")
