import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

class ARIMABaseline:
    """
    ARIMA baseline model for time series forecasting
    """
    
    def __init__(self, max_p=5, max_d=2, max_q=5):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.aic_score = np.inf
        
    def check_stationarity(self, data, alpha=0.05):
        """
        Check if the time series is stationary using Augmented Dickey-Fuller test
        
        Args:
            data: Time series data
            alpha: Significance level
            
        Returns:
            bool: True if stationary, False otherwise
        """
        try:
            result = adfuller(data, autolag='AIC')
            p_value = result[1]
            return p_value < alpha
        except:
            return False
    
    def difference_data(self, data, max_diff=2):
        """
        Apply differencing to make the series stationary
        
        Args:
            data: Time series data
            max_diff: Maximum number of differencing operations
            
        Returns:
            differenced_data, d: Differenced data and number of differences applied
        """
        d = 0
        current_data = data.copy()
        
        for i in range(max_diff):
            if self.check_stationarity(current_data):
                break
            current_data = np.diff(current_data)
            d += 1
        
        return current_data, d
    
    def auto_arima(self, data, seasonal=False, stepwise=True):
        """
        Automatic ARIMA model selection using grid search
        
        Args:
            data: Time series data
            seasonal: Whether to include seasonal components
            stepwise: Whether to use stepwise selection
            
        Returns:
            best_params: Best (p, d, q) parameters
        """
        # Check stationarity and determine d
        differenced_data, d = self.difference_data(data, self.max_d)
        
        best_aic = np.inf
        best_params = (0, d, 0)
        
        # Grid search for p and q
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                try:
                    # Fit ARIMA model
                    model = ARIMA(data, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    # Check AIC
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        
                except:
                    continue
        
        self.best_params = best_params
        self.aic_score = best_aic
        
        return best_params
    
    def fit(self, data, order=None):
        """
        Fit ARIMA model to the data
        
        Args:
            data: Time series data
            order: (p, d, q) parameters. If None, auto-select
            
        Returns:
            fitted_model: Fitted ARIMA model
        """
        if order is None:
            order = self.auto_arima(data)
        
        try:
            self.model = ARIMA(data, order=order)
            self.fitted_model = self.model.fit()
            return self.fitted_model
        
        except Exception as e:
            # Fallback to simple ARIMA(1,1,1)
            try:
                self.model = ARIMA(data, order=(1, 1, 1))
                self.fitted_model = self.model.fit()
                return self.fitted_model
            except:
                # Final fallback to random walk
                self.model = ARIMA(data, order=(0, 1, 0))
                self.fitted_model = self.model.fit()
                return self.fitted_model
    
    def predict(self, steps):
        """
        Generate forecasts
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            predictions: Forecasted values
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            return forecast.values if hasattr(forecast, 'values') else forecast
        
        except Exception as e:
            # Fallback: repeat last known value
            last_value = self.fitted_model.fittedvalues.iloc[-1]
            return np.full(steps, last_value)
    
    def fit_predict(self, data, forecast_horizon, order=None):
        """
        Fit model and generate predictions in one step
        
        Args:
            data: Time series data
            forecast_horizon: Number of steps to forecast
            order: ARIMA order (p, d, q)
            
        Returns:
            predictions: Forecasted values
        """
        # Ensure we have enough data
        if len(data) < 10:
            # Not enough data for ARIMA, return mean
            return np.full(forecast_horizon, np.mean(data))
        
        # Fit the model
        self.fit(data, order)
        
        # Generate predictions
        predictions = self.predict(forecast_horizon)
        
        return predictions
    
    def get_model_summary(self):
        """Get model summary and diagnostics"""
        if self.fitted_model is None:
            return None
        
        try:
            summary = {
                'order': self.fitted_model.model.order,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'hqic': self.fitted_model.hqic,
                'llf': self.fitted_model.llf,
                'params': self.fitted_model.params.to_dict() if hasattr(self.fitted_model.params, 'to_dict') else {}
            }
            return summary
        except:
            return {'order': (1, 1, 1), 'aic': np.inf}
    
    def get_confidence_intervals(self, steps, alpha=0.05):
        """
        Get confidence intervals for predictions
        
        Args:
            steps: Number of forecast steps
            alpha: Significance level (1-alpha = confidence level)
            
        Returns:
            lower_bounds, upper_bounds: Confidence interval bounds
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        try:
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            conf_int = forecast_result.conf_int(alpha=alpha)
            
            if hasattr(conf_int, 'values'):
                lower_bounds = conf_int.values[:, 0]
                upper_bounds = conf_int.values[:, 1]
            else:
                lower_bounds = conf_int.iloc[:, 0].values
                upper_bounds = conf_int.iloc[:, 1].values
            
            return lower_bounds, upper_bounds
            
        except Exception as e:
            # Fallback: use simple standard deviation estimate
            residuals = self.fitted_model.resid
            std_error = np.std(residuals)
            predictions = self.predict(steps)
            
            # Simple confidence intervals
            margin = 1.96 * std_error  # 95% confidence
            lower_bounds = predictions - margin
            upper_bounds = predictions + margin
            
            return lower_bounds, upper_bounds
    
    def calculate_residuals(self):
        """Calculate model residuals"""
        if self.fitted_model is None:
            return None
        
        try:
            return self.fitted_model.resid.values
        except:
            return np.array([])
    
    def validate_model(self, data, train_ratio=0.8):
        """
        Validate model using train-test split
        
        Args:
            data: Complete time series data
            train_ratio: Ratio of data to use for training
            
        Returns:
            metrics: Dictionary with validation metrics
        """
        # Split data
        split_point = int(len(data) * train_ratio)
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        if len(train_data) < 10 or len(test_data) < 1:
            return {'rmse': np.inf, 'mae': np.inf, 'mape': np.inf}
        
        # Fit on training data
        self.fit(train_data)
        
        # Predict test period
        predictions = self.predict(len(test_data))
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((test_data - predictions) ** 2))
        mae = np.mean(np.abs(test_data - predictions))
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'predictions': predictions,
            'actual': test_data
        }
