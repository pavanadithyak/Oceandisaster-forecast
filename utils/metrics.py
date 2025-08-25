import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ModelMetrics:
    """
    Comprehensive metrics calculation for time series forecasting models
    """
    
    def __init__(self):
        pass
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with various metrics
        """
        # Ensure arrays are the same length
        min_length = min(len(y_true), len(y_pred))
        y_true = np.array(y_true)[:min_length]
        y_pred = np.array(y_pred)[:min_length]
        
        # Handle edge cases
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                'mse': np.inf,
                'rmse': np.inf,
                'mae': np.inf,
                'mape': np.inf,
                'r2': -np.inf,
                'adjusted_r2': -np.inf,
                'explained_variance': 0,
                'max_error': np.inf,
                'mean_error': np.inf
            }
        
        try:
            # Basic metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Mean Absolute Percentage Error
            mape = self.calculate_mape(y_true, y_pred)
            
            # Adjusted R²
            n = len(y_true)
            p = 1  # Number of predictors (assuming single feature)
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else r2
            
            # Explained Variance Score
            explained_variance = self.calculate_explained_variance(y_true, y_pred)
            
            # Error statistics
            errors = y_true - y_pred
            max_error = np.max(np.abs(errors))
            mean_error = np.mean(errors)
            
            # Additional metrics
            median_absolute_error = np.median(np.abs(errors))
            
            return {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'r2': float(r2),
                'adjusted_r2': float(adjusted_r2),
                'explained_variance': float(explained_variance),
                'max_error': float(max_error),
                'mean_error': float(mean_error),
                'median_absolute_error': float(median_absolute_error)
            }
            
        except Exception as e:
            # Return default values on error
            return {
                'mse': np.inf,
                'rmse': np.inf,
                'mae': np.inf,
                'mape': np.inf,
                'r2': -np.inf,
                'adjusted_r2': -np.inf,
                'explained_variance': 0,
                'max_error': np.inf,
                'mean_error': np.inf,
                'median_absolute_error': np.inf
            }
    
    def calculate_mape(self, y_true, y_pred, epsilon=1e-8):
        """
        Calculate Mean Absolute Percentage Error
        
        Args:
            y_true: True values
            y_pred: Predicted values
            epsilon: Small value to avoid division by zero
            
        Returns:
            MAPE value
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Avoid division by zero
        denominator = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100
        
        return mape
    
    def calculate_explained_variance(self, y_true, y_pred):
        """
        Calculate explained variance score
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Explained variance score
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        var_y = np.var(y_true)
        var_diff = np.var(y_true - y_pred)
        
        if var_y == 0:
            return 0.0
        
        return 1 - (var_diff / var_y)
    
    def calculate_directional_accuracy(self, y_true, y_pred):
        """
        Calculate directional accuracy (percentage of correct direction predictions)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Directional accuracy percentage
        """
        if len(y_true) < 2 or len(y_pred) < 2:
            return 0.0
        
        true_directions = np.diff(y_true) > 0
        pred_directions = np.diff(y_pred) > 0
        
        correct_directions = true_directions == pred_directions
        directional_accuracy = np.mean(correct_directions) * 100
        
        return directional_accuracy
    
    def calculate_forecast_bias(self, y_true, y_pred):
        """
        Calculate forecast bias metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with bias metrics
        """
        errors = np.array(y_pred) - np.array(y_true)
        
        # Mean Bias Error
        mbe = np.mean(errors)
        
        # Mean Absolute Bias
        mab = np.mean(np.abs(errors))
        
        # Bias percentage
        mean_true = np.mean(y_true)
        bias_percentage = (mbe / mean_true) * 100 if mean_true != 0 else 0
        
        # Theil's U statistic
        theil_u = self.calculate_theil_u(y_true, y_pred)
        
        return {
            'mean_bias_error': float(mbe),
            'mean_absolute_bias': float(mab),
            'bias_percentage': float(bias_percentage),
            'theil_u': float(theil_u)
        }
    
    def calculate_theil_u(self, y_true, y_pred):
        """
        Calculate Theil's U statistic for forecast accuracy
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Theil's U statistic
        """
        if len(y_true) < 2:
            return np.inf
        
        # Calculate relative changes
        true_changes = np.diff(y_true) / y_true[:-1]
        pred_changes = np.diff(y_pred) / y_true[:-1]  # Use true values in denominator
        
        # Handle division by zero
        true_changes = np.where(np.isfinite(true_changes), true_changes, 0)
        pred_changes = np.where(np.isfinite(pred_changes), pred_changes, 0)
        
        # Calculate Theil's U
        numerator = np.sqrt(np.mean((pred_changes - true_changes) ** 2))
        denominator = np.sqrt(np.mean(true_changes ** 2))
        
        if denominator == 0:
            return np.inf
        
        return numerator / denominator
    
    def calculate_interval_score(self, y_true, lower_bound, upper_bound, alpha=0.05):
        """
        Calculate interval score for prediction intervals
        
        Args:
            y_true: True values
            lower_bound: Lower bound of prediction interval
            upper_bound: Upper bound of prediction interval
            alpha: Significance level (1-alpha = confidence level)
            
        Returns:
            Interval score
        """
        y_true = np.array(y_true)
        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)
        
        # Interval width
        width = upper_bound - lower_bound
        
        # Penalties for values outside the interval
        lower_penalty = (2 / alpha) * (lower_bound - y_true) * (y_true < lower_bound)
        upper_penalty = (2 / alpha) * (y_true - upper_bound) * (y_true > upper_bound)
        
        # Interval score
        interval_score = width + lower_penalty + upper_penalty
        
        return np.mean(interval_score)
    
    def calculate_coverage_probability(self, y_true, lower_bound, upper_bound):
        """
        Calculate coverage probability of prediction intervals
        
        Args:
            y_true: True values
            lower_bound: Lower bound of prediction interval
            upper_bound: Upper bound of prediction interval
            
        Returns:
            Coverage probability (percentage)
        """
        y_true = np.array(y_true)
        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)
        
        # Check if true values are within the intervals
        within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        coverage = np.mean(within_interval) * 100
        
        return coverage
    
    def comprehensive_evaluation(self, y_true, y_pred, lower_bound=None, upper_bound=None):
        """
        Perform comprehensive model evaluation
        
        Args:
            y_true: True values
            y_pred: Predicted values
            lower_bound: Lower bound of prediction intervals (optional)
            upper_bound: Upper bound of prediction intervals (optional)
            
        Returns:
            Dictionary with all metrics
        """
        # Basic metrics
        basic_metrics = self.calculate_metrics(y_true, y_pred)
        
        # Bias metrics
        bias_metrics = self.calculate_forecast_bias(y_true, y_pred)
        
        # Directional accuracy
        directional_acc = self.calculate_directional_accuracy(y_true, y_pred)
        
        # Combine all metrics
        all_metrics = {
            **basic_metrics,
            **bias_metrics,
            'directional_accuracy': directional_acc
        }
        
        # Add interval metrics if bounds are provided
        if lower_bound is not None and upper_bound is not None:
            interval_score = self.calculate_interval_score(y_true, lower_bound, upper_bound)
            coverage_prob = self.calculate_coverage_probability(y_true, lower_bound, upper_bound)
            
            all_metrics.update({
                'interval_score': interval_score,
                'coverage_probability': coverage_prob
            })
        
        return all_metrics
    
    def format_metrics_display(self, metrics):
        """
        Format metrics for display
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Formatted string
        """
        display_lines = []
        display_lines.append("Model Performance Metrics:")
        display_lines.append("-" * 30)
        
        # Main metrics
        if 'r2' in metrics:
            display_lines.append(f"R² Score: {metrics['r2']:.4f}")
        if 'rmse' in metrics:
            display_lines.append(f"RMSE: {metrics['rmse']:.4f}")
        if 'mae' in metrics:
            display_lines.append(f"MAE: {metrics['mae']:.4f}")
        if 'mape' in metrics:
            display_lines.append(f"MAPE: {metrics['mape']:.2f}%")
        
        # Additional metrics
        if 'directional_accuracy' in metrics:
            display_lines.append(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        
        if 'coverage_probability' in metrics:
            display_lines.append(f"Coverage Probability: {metrics['coverage_probability']:.2f}%")
        
        return "\n".join(display_lines)
