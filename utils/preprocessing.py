import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesPreprocessor:
    """
    Comprehensive preprocessing utilities for time series data
    """
    
    def __init__(self, scaling_method='minmax', imputation_strategy='linear'):
        """
        Initialize preprocessor
        
        Args:
            scaling_method: 'minmax', 'standard', or 'robust'
            imputation_strategy: 'linear', 'mean', 'median', 'forward_fill'
        """
        self.scaling_method = scaling_method
        self.imputation_strategy = imputation_strategy
        self.scalers = {}
        self.imputers = {}
        
    def handle_missing_values(self, data, columns=None):
        """
        Handle missing values in the dataset
        
        Args:
            data: DataFrame with missing values
            columns: List of columns to process (None for all numeric columns)
            
        Returns:
            DataFrame with missing values handled
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        processed_data = data.copy()
        
        for col in columns:
            if col not in processed_data.columns:
                continue
                
            if processed_data[col].isna().any():
                if self.imputation_strategy == 'linear':
                    # Linear interpolation
                    processed_data[col] = processed_data[col].interpolate(
                        method='linear', limit_direction='both'
                    )
                    
                elif self.imputation_strategy == 'forward_fill':
                    # Forward fill then backward fill
                    processed_data[col] = processed_data[col].fillna(method='ffill').fillna(method='bfill')
                    
                elif self.imputation_strategy in ['mean', 'median']:
                    # Statistical imputation
                    if col not in self.imputers:
                        self.imputers[col] = SimpleImputer(strategy=self.imputation_strategy)
                        self.imputers[col].fit(processed_data[[col]])
                    
                    imputed_values = self.imputers[col].transform(processed_data[[col]])
                    processed_data[col] = imputed_values.flatten()
                
                # Fill any remaining NaN values with column mean
                if processed_data[col].isna().any():
                    processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
        
        return processed_data
    
    def detect_outliers(self, data, columns=None, method='iqr', threshold=1.5):
        """
        Detect outliers in the data
        
        Args:
            data: DataFrame to analyze
            columns: Columns to check for outliers
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier flags
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        outlier_flags = pd.DataFrame(index=data.index)
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_flags[f'{col}_outlier'] = (data[col] < lower_bound) | (data[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_flags[f'{col}_outlier'] = z_scores > threshold
        
        return outlier_flags
    
    def handle_outliers(self, data, outlier_flags, method='cap'):
        """
        Handle outliers in the data
        
        Args:
            data: DataFrame with outliers
            outlier_flags: DataFrame with outlier flags
            method: 'cap', 'remove', or 'interpolate'
            
        Returns:
            DataFrame with outliers handled
        """
        processed_data = data.copy()
        
        for col in data.select_dtypes(include=[np.number]).columns:
            outlier_col = f'{col}_outlier'
            if outlier_col not in outlier_flags.columns:
                continue
            
            outlier_mask = outlier_flags[outlier_col]
            
            if method == 'cap':
                # Cap outliers to 5th and 95th percentiles
                lower_cap = data[col].quantile(0.05)
                upper_cap = data[col].quantile(0.95)
                processed_data.loc[outlier_mask, col] = processed_data.loc[outlier_mask, col].clip(
                    lower=lower_cap, upper=upper_cap
                )
                
            elif method == 'interpolate':
                # Replace outliers with interpolated values
                processed_data.loc[outlier_mask, col] = np.nan
                processed_data[col] = processed_data[col].interpolate(method='linear')
                
            elif method == 'remove':
                # Remove outlier rows (use with caution for time series)
                processed_data = processed_data[~outlier_mask]
        
        return processed_data
    
    def scale_data(self, data, columns=None, fit=True):
        """
        Scale the data using the specified scaling method
        
        Args:
            data: DataFrame to scale
            columns: Columns to scale (None for all numeric columns)
            fit: Whether to fit the scaler (True for training, False for test)
            
        Returns:
            Scaled DataFrame
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        scaled_data = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
            
            if fit or col not in self.scalers:
                # Create and fit scaler
                if self.scaling_method == 'minmax':
                    self.scalers[col] = MinMaxScaler()
                elif self.scaling_method == 'standard':
                    self.scalers[col] = StandardScaler()
                elif self.scaling_method == 'robust':
                    self.scalers[col] = RobustScaler()
                
                # Fit scaler
                self.scalers[col].fit(data[[col]])
            
            # Transform data
            scaled_values = self.scalers[col].transform(data[[col]])
            scaled_data[col] = scaled_values.flatten()
        
        return scaled_data
    
    def inverse_scale(self, data, columns=None):
        """
        Inverse transform scaled data
        
        Args:
            data: Scaled data to inverse transform
            columns: Columns to inverse transform
            
        Returns:
            Original scale data
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        inverse_data = data.copy()
        
        for col in columns:
            if col in self.scalers and col in data.columns:
                if isinstance(data[col], pd.Series):
                    inverse_values = self.scalers[col].inverse_transform(data[[col]])
                    inverse_data[col] = inverse_values.flatten()
                else:
                    # Handle numpy arrays
                    inverse_values = self.scalers[col].inverse_transform(data[col].reshape(-1, 1))
                    inverse_data[col] = inverse_values.flatten()
        
        return inverse_data
    
    def create_sequences(self, data, sequence_length, target_column=None, stride=1):
        """
        Create sequences for time series modeling
        
        Args:
            data: Time series data (array or DataFrame)
            sequence_length: Length of input sequences
            target_column: Target column name (for DataFrame input)
            stride: Step size between sequences
            
        Returns:
            sequences, targets: Input sequences and target values
        """
        if isinstance(data, pd.DataFrame):
            if target_column is None:
                # Use the first numeric column as target
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                target_column = numeric_cols[0] if len(numeric_cols) > 0 else data.columns[0]
            
            values = data[target_column].values
        else:
            values = np.array(data)
        
        sequences = []
        targets = []
        
        for i in range(0, len(values) - sequence_length, stride):
            # Input sequence
            seq = values[i:i + sequence_length]
            # Target (next value)
            target = values[i + sequence_length]
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def add_time_features(self, data, time_column='time'):
        """
        Add time-based features to the dataset
        
        Args:
            data: DataFrame with time column
            time_column: Name of the time column
            
        Returns:
            DataFrame with additional time features
        """
        enhanced_data = data.copy()
        
        if time_column not in data.columns:
            return enhanced_data
        
        # Ensure datetime format
        enhanced_data[time_column] = pd.to_datetime(enhanced_data[time_column])
        
        # Extract time features
        enhanced_data['hour'] = enhanced_data[time_column].dt.hour
        enhanced_data['day_of_week'] = enhanced_data[time_column].dt.dayofweek
        enhanced_data['month'] = enhanced_data[time_column].dt.month
        enhanced_data['day_of_year'] = enhanced_data[time_column].dt.dayofyear
        
        # Cyclical encoding for periodic features
        enhanced_data['hour_sin'] = np.sin(2 * np.pi * enhanced_data['hour'] / 24)
        enhanced_data['hour_cos'] = np.cos(2 * np.pi * enhanced_data['hour'] / 24)
        enhanced_data['day_sin'] = np.sin(2 * np.pi * enhanced_data['day_of_week'] / 7)
        enhanced_data['day_cos'] = np.cos(2 * np.pi * enhanced_data['day_of_week'] / 7)
        enhanced_data['month_sin'] = np.sin(2 * np.pi * enhanced_data['month'] / 12)
        enhanced_data['month_cos'] = np.cos(2 * np.pi * enhanced_data['month'] / 12)
        
        return enhanced_data
    
    def preprocess(self, data, fit=True, handle_outliers=True):
        """
        Complete preprocessing pipeline
        
        Args:
            data: Raw data to preprocess
            fit: Whether to fit preprocessing components
            handle_outliers: Whether to detect and handle outliers
            
        Returns:
            Preprocessed data
        """
        # Copy data
        processed_data = data.copy()
        
        # Handle missing values
        processed_data = self.handle_missing_values(processed_data)
        
        # Add time features if time column exists
        if 'time' in processed_data.columns:
            processed_data = self.add_time_features(processed_data)
        
        # Handle outliers
        if handle_outliers:
            outlier_flags = self.detect_outliers(processed_data)
            if not outlier_flags.empty:
                processed_data = self.handle_outliers(processed_data, outlier_flags, method='cap')
        
        # Scale numeric data
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        exclude_columns = ['hour', 'day_of_week', 'month', 'day_of_year']  # Don't scale discrete features
        scale_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        if scale_columns:
            processed_data = self.scale_data(processed_data, columns=scale_columns, fit=fit)
        
        return processed_data
    
    def get_preprocessing_info(self):
        """Get information about fitted preprocessing components"""
        info = {
            'scaling_method': self.scaling_method,
            'imputation_strategy': self.imputation_strategy,
            'fitted_scalers': list(self.scalers.keys()),
            'fitted_imputers': list(self.imputers.keys())
        }
        return info
