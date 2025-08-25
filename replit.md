# Ocean Forecasting Dashboard

## Overview

This is a comprehensive ocean forecasting dashboard built with Streamlit that provides real-time marine data visualization and predictive modeling capabilities. The application fetches marine and weather data from the Open-Meteo Marine API and uses advanced machine learning models (Bayesian LSTM and ARIMA) to forecast ocean conditions and assess environmental risks. The dashboard serves maritime professionals, researchers, and coastal communities by providing actionable insights about sea level variations, wave conditions, wind patterns, and associated risks.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **Visualization**: Plotly for interactive charts and graphs with subplots support
- **Styling**: Custom CSS for enhanced UI components including metric cards and risk alerts
- **Layout**: Wide layout with expandable sidebar for configuration options

### Backend Architecture
- **Data Processing**: Pandas and NumPy for data manipulation and numerical computations
- **Machine Learning**: PyTorch-based Bayesian LSTM with Monte Carlo Dropout for uncertainty quantification
- **Statistical Modeling**: ARIMA baseline model using statsmodels for time series forecasting
- **Preprocessing Pipeline**: Comprehensive data cleaning, scaling (MinMax/Standard/Robust), and missing value imputation

### Core Components
1. **Data Layer**: API client for fetching real-time marine data with session management and error handling
2. **Model Layer**: Dual modeling approach with Bayesian neural networks and traditional statistical methods
3. **Preprocessing Layer**: Automated data cleaning, feature scaling, and time series preparation
4. **Risk Assessment**: Multi-parameter risk evaluation with customizable thresholds and alert systems
5. **Metrics Layer**: Comprehensive model evaluation including MSE, RMSE, MAE, MAPE, and R² metrics

### Design Patterns
- **Modular Architecture**: Clear separation of concerns with dedicated modules for data, models, and utilities
- **Object-Oriented Design**: Class-based structure for reusable components and state management
- **Configuration-Driven**: Parameterized models and preprocessing with flexible threshold settings
- **Error Handling**: Robust exception handling throughout the data pipeline and model training

## External Dependencies

### APIs and Data Sources
- **Open-Meteo Marine API**: Primary data source for marine conditions, wave heights, wind patterns, and atmospheric data
- **Marine Parameters**: Wave height/direction/period, swell data, wind wave measurements
- **Weather Parameters**: Temperature, wind speed/direction, atmospheric pressure

### Machine Learning Libraries
- **PyTorch**: Deep learning framework for Bayesian LSTM implementation with CUDA support
- **scikit-learn**: Data preprocessing, scaling, imputation, and model evaluation metrics
- **statsmodels**: ARIMA modeling, stationarity testing, and time series decomposition

### Visualization and Web Framework
- **Streamlit**: Web application framework with built-in state management
- **Plotly**: Interactive plotting library with support for real-time updates and subplots
- **Pandas/NumPy**: Data manipulation and numerical computing foundations

### Additional Dependencies
- **requests**: HTTP client for API interactions with session management
- **datetime**: Time series data handling and date range calculations
- **warnings**: Error suppression for cleaner user experience