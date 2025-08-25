import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time

class MarineDataClient:
    """Client for fetching marine data from Open-Meteo Marine API"""
    
    def __init__(self):
        self.base_url = "https://marine-api.open-meteo.com/v1/marine"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Ocean-Forecasting-Dashboard/1.0'
        })
        
    def get_marine_data(self, latitude, longitude, days=7):
        """
        Fetch marine data for a specific location
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            days (int): Number of days to fetch data for
            
        Returns:
            pd.DataFrame: Marine data with timestamps
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Parameters to fetch
        marine_params = [
            "wave_height",
            "wave_direction", 
            "wave_period",
            "wind_wave_height",
            "wind_wave_direction",
            "wind_wave_period",
            "swell_wave_height",
            "swell_wave_direction",
            "swell_wave_period"
        ]
        
        weather_params = [
            "temperature_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "pressure_msl"
        ]
        
        try:
            # Fetch marine data
            marine_response = self._make_request(
                latitude, longitude, start_date, end_date, marine_params, "marine"
            )
            
            # Fetch weather data
            weather_response = self._make_request(
                latitude, longitude, start_date, end_date, weather_params, "weather"
            )
            
            # Process and combine data
            marine_data = self._process_response(marine_response)
            weather_data = self._process_response(weather_response)
            
            # Merge data on time
            combined_data = pd.merge(marine_data, weather_data, on='time', how='outer')
            
            # Add synthetic sea level data (since not directly available)
            combined_data['sea_level'] = self._calculate_sea_level(combined_data)
            
            # Sort by time and reset index
            combined_data = combined_data.sort_values('time').reset_index(drop=True)
            
            # Fill missing values
            combined_data = self._handle_missing_values(combined_data)
            
            return combined_data
            
        except Exception as e:
            raise Exception(f"Failed to fetch marine data: {str(e)}")
    
    def _make_request(self, lat, lon, start_date, end_date, params, data_type):
        """Make API request with retry logic"""
        
        if data_type == "marine":
            url = self.base_url
        else:
            url = "https://api.open-meteo.com/v1/forecast"
        
        query_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": ",".join(params),
            "timezone": "UTC"
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=query_params, timeout=30)
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _process_response(self, response_data):
        """Process API response into DataFrame"""
        try:
            hourly_data = response_data.get('hourly', {})
            
            if not hourly_data:
                raise ValueError("No hourly data in response")
            
            # Extract time series
            times = hourly_data.get('time', [])
            
            if not times:
                raise ValueError("No time data in response")
            
            # Create DataFrame
            df = pd.DataFrame({'time': pd.to_datetime(times)})
            
            # Add all available parameters
            for key, values in hourly_data.items():
                if key != 'time' and values:
                    df[key] = values
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to process response: {str(e)}")
    
    def _calculate_sea_level(self, data):
        """
        Calculate synthetic sea level based on atmospheric pressure and wind
        This is a simplified model for demonstration
        """
        # Initialize sea level with small random variations
        base_level = np.random.normal(0, 0.1, len(data))
        
        # Add pressure effect (inverse relationship)
        if 'pressure_msl' in data.columns:
            pressure_norm = (data['pressure_msl'] - 1013.25) / 50  # Normalize around standard pressure
            base_level -= pressure_norm * 0.3  # Inverse relationship
        
        # Add wind effect
        if 'wind_speed_10m' in data.columns:
            wind_effect = data['wind_speed_10m'] * 0.02  # Wind piles up water
            base_level += wind_effect
        
        # Add wave height effect if available
        if 'wave_height' in data.columns:
            wave_effect = data['wave_height'] * 0.1
            base_level += wave_effect
        
        # Add tidal-like pattern (simplified)
        time_hours = np.arange(len(data))
        tidal_pattern = 0.5 * np.sin(2 * np.pi * time_hours / 12.42)  # Semi-diurnal tide
        base_level += tidal_pattern
        
        return base_level
    
    def _handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        # Forward fill then backward fill
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # For any remaining NaN values, use interpolation
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].interpolate(method='linear')
        
        # If still NaN, fill with column mean
        for col in numeric_columns:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())
        
        return data
    
    def get_current_conditions(self, latitude, longitude):
        """Get current marine conditions"""
        try:
            # Get data for the last 24 hours
            data = self.get_marine_data(latitude, longitude, days=1)
            
            if data.empty:
                return None
            
            # Return the most recent data point
            return data.iloc[-1].to_dict()
            
        except Exception as e:
            raise Exception(f"Failed to get current conditions: {str(e)}")
    
    def validate_coordinates(self, latitude, longitude):
        """Validate coordinate ranges"""
        if not (-90 <= latitude <= 90):
            raise ValueError("Latitude must be between -90 and 90 degrees")
        
        if not (-180 <= longitude <= 180):
            raise ValueError("Longitude must be between -180 and 180 degrees")
        
        return True
