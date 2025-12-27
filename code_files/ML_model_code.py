"""
-----------------------------------------------------------------------------
ML Model FOR THE PROJECT CLIMATE CHANGE (PROTOTYPE)
-----------------------------------------------------------------------------
NOTE : this is the prototype ML model that predicts the whether AQI prediction
       this model do prediction for 24 hours and also 7 days prediction by 
       seeing user location coordinates and then gives the prediction of 
       Aqi of the user's location 

Result format : 
=== FINAL RESULTS (Daily Average AQI) ===
Random Forest MAE: 37.4594

=== 24-HOUR HOURLY FORECAST ===
            Time  Predicted AQI                         Status
2025-12-27 14:00          97.67                       Moderate
2025-12-27 15:00         102.81 Unhealthy for Sensitive Groups
2025-12-27 16:00         113.09 Unhealthy for Sensitive Groups
2025-12-27 17:00         123.38 Unhealthy for Sensitive Groups
2025-12-27 18:00         133.66 Unhealthy for Sensitive Groups
2025-12-27 19:00         128.52 Unhealthy for Sensitive Groups
2025-12-27 20:00         118.23 Unhealthy for Sensitive Groups
2025-12-27 21:00         102.81 Unhealthy for Sensitive Groups
2025-12-27 22:00          92.53                       Moderate
2025-12-27 23:00          87.39                       Moderate
2025-12-28 00:00          82.25                       Moderate
2025-12-28 01:00          77.11                       Moderate
2025-12-28 02:00          71.97                       Moderate
2025-12-28 03:00          71.97                       Moderate
2025-12-28 04:00          77.11                       Moderate
2025-12-28 05:00          87.39                       Moderate
2025-12-28 06:00         102.81 Unhealthy for Sensitive Groups
2025-12-28 07:00         123.38 Unhealthy for Sensitive Groups
2025-12-28 08:00         133.66 Unhealthy for Sensitive Groups
2025-12-28 09:00         123.38 Unhealthy for Sensitive Groups
2025-12-28 10:00         113.09 Unhealthy for Sensitive Groups
2025-12-28 11:00         102.81 Unhealthy for Sensitive Groups
2025-12-28 12:00          92.53                       Moderate
2025-12-28 13:00          92.53                       Moderate

=== 7-DAY AQI FORECAST ===
  Day       Date                    Location  Latitude  Longitude  Predicted AQI                         Status        
Day-1 2026-01-03 Maninagar, Ahmedabad, India 23.002657  72.591912         102.17 Unhealthy for Sensitive Groups        
Day-2 2026-01-04 Maninagar, Ahmedabad, India 23.002657  72.591912          93.87                       Moderate        
Day-3 2026-01-05 Maninagar, Ahmedabad, India 23.002657  72.591912          95.32                       Moderate        
Day-4 2026-01-06 Maninagar, Ahmedabad, India 23.002657  72.591912          87.71                       Moderate        
Day-5 2026-01-07 Maninagar, Ahmedabad, India 23.002657  72.591912          85.53                       Moderate        
Day-6 2026-01-08 Maninagar, Ahmedabad, India 23.002657  72.591912          87.83                       Moderate        
Day-7 2026-01-09 Maninagar, Ahmedabad, India 23.002657  72.591912          87.74                       Moderate        
==========================
"""

import requests
# Standard Data Science & Plots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Scikit-Learn (Machine Learning)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

# TensorFlow / Keras (Deep Learning) - Optional
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not found. LSTM model will be disabled.")
    TF_AVAILABLE = False

# Geocoder (Location Detection) - Optional
try:
    import geocoder
    GEOCODER_AVAILABLE = True
except ImportError:
    print("Warning: 'geocoder' library not found. Falling back to API-based auto-location.")
    GEOCODER_AVAILABLE = False

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
API_KEY = "24da0fb52646fa4b9ad08d97d043ea86d2c31983"
# Base URL will be determined dynamically
DEFAULT_BASE_URL = "https://api.waqi.info/feed/here/"

class AQIDataFetcher:
    """
    Handles fetching data from WAQI API.
    """
    def __init__(self, api_token, lat=None, lon=None):
        self.api_token = api_token
        if lat and lon:
            self.base_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/"
            print(f"Targeting API for specific coordinates: {lat}, {lon}")
        else:
            self.base_url = DEFAULT_BASE_URL
            print("Targeting API for auto-detected IP location (feed/here/).")

    def fetch_data(self):
        """
        Fetches current data and daily forecast.
        Returns:
            df (pd.DataFrame): Daily forecast + history values.
            metadata (dict): Location info (city, lat, lon).
        """
        # Prepare parameters for the API request
        params = {'token': self.api_token}
        
        print(f"Fetching data from WAQI API...")
        try:
            # Make the HTTP GET request
            response = requests.get(self.base_url, params=params)
            response.raise_for_status() # Check for HTTP errors
            data = response.json()
            
            # Validate API response status
            if data['status'] != 'ok':
                raise ValueError(f"API Error: {data.get('data', 'Unknown error')}")

            # Extract Daily Forecast for PM2.5 from the complex JSON structure
            forecast_data = data['data'].get('forecast', {}).get('daily', {}).get('pm25', [])
            
            # Extract Location Metadata to display to the user
            city_info = data['data'].get('city', {})
            metadata = {
                'city': city_info.get('name', 'Unknown'),
                'lat': city_info.get('geo', [0, 0])[0], 
                'lon': city_info.get('geo', [0, 0])[1]
            }
            
            if not forecast_data:
                print("No forecast data found in API response.")
                return pd.DataFrame(), metadata

            # Convert JSON list to DataFrame suitable for analysis
            records = []
            for item in forecast_data:
                records.append({
                    'dt': pd.to_datetime(item['day']), # Convert string date to datetime object
                    'aqi': item['avg'], 
                    'min': item['min'],
                    'max': item['max']
                })
            
            df = pd.DataFrame(records)
            df.set_index('dt', inplace=True) # Set date as the index for time-series operations
            df.sort_index(inplace=True)      # Ensure data is chronologically sorted
            
            # Get Current 'Real' AQI for immediate display
            current_aqi = data['data'].get('aqi')
            print(f"Current Real-Time AQI: {current_aqi}")
            print(f"Confirmed Station: {metadata['city']} (Lat: {metadata['lat']}, Lon: {metadata['lon']})")
            
            return df, metadata

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame(), {}

class DataPreprocessor:
    """
    Handles data cleaning, filling missing values, and feature engineering.
    """
    def preprocess_and_augment(self, df):
        """
        1. Preprocesses the real data.
        2. AUGMENTS it with synthetic history because 7-10 days is NOT enough for training.
        """
        if df.empty:
            print("DataFrame is empty. Skipping preprocessing.")
            return df

        print("Preprocessing data...")
        
        # 1. Augmentation (Create Synthetic History for Training)
        # We need at least ~100 points to train a reliable model, but the API often provides only ~7-10 days.
        # We'll simulate the last 60 days using the statistical properties (mean, std) of the fetched data.
        real_mean = df['aqi'].mean()
        real_std = df['aqi'].std() if len(df) > 1 else 10
        
        if np.isnan(real_std): real_std = 10
        
        last_date = df.index[-1]
        start_date = last_date - timedelta(days=60)
        synthetic_dates = pd.date_range(start=start_date, end=last_date - timedelta(days=1), freq='D')
        
        # Generate synthetic AQI values with some random walk / seasonality
        # This creates a realistic-looking history for the model to learn patterns from.
        synthetic_aqi = []
        val = real_mean
        for _ in range(len(synthetic_dates)):
            val += np.random.normal(0, real_std * 0.5) # Add random noise (Random walk)
            val = max(10, min(500, val))               # Clip to realistic AQI bounds (0-500)
            synthetic_aqi.append(val)
            
        df_synthetic = pd.DataFrame({
            'aqi': synthetic_aqi,
            'min': [v - 10 for v in synthetic_aqi], # Heuristic min
            'max': [v + 10 for v in synthetic_aqi]  # Heuristic max
        }, index=synthetic_dates)
        
        # Combine Synthetic + Real Data
        # We prioritize real data where overlap exists (though here we designed no overlap)
        df_final = pd.concat([df_synthetic, df])
        df_final.sort_index(inplace=True)
        # Remove duplicates index
        df_final = df_final[~df_final.index.duplicated(keep='last')]
        
        print(f"Augmented data size: {len(df_final)} records (Synthetic + Real)")

        # 2. Feature Engineering
        # Date Features: Capture weekly and monthly seasonality
        df_final['day_of_week'] = df_final.index.dayofweek
        df_final['month'] = df_final.index.month
        
        # Cyclical Features: Encode day of week as (sin, cos) to preserve cyclical continuity
        # (e.g., Sunday is close to Monday)
        df_final['day_sin'] = np.sin(2 * np.pi * df_final['day_of_week'] / 7)
        df_final['day_cos'] = np.cos(2 * np.pi * df_final['day_of_week'] / 7)

        # 3. Create Lag Features (Past Days)
        # Use previous days' AQI to predict the next day's AQI (Autoregressive approach)
        target_col = 'aqi'
        for lag in [1, 2, 3]: # Lag of 1 day, 2 days, 3 days
            df_final[f'lag_{lag}d'] = df_final[target_col].shift(lag)

        # Drop rows with NaN values created by shifting (the first few rows)
        df_final.dropna(inplace=True)
        
        print(f"Data shape after preprocessing: {df_final.shape}")
        return df_final

class AQIModels:
    """
    Contains Model definitions, training logic, and evaluation.
    """
    
    def train_random_forest(self, df, target_col='aqi'):
        """
        Trains a Random Forest Regressor.
        """
        print("\n--- Training Random Forest Regressor ---")
        
        # Prepare Feature Matrix (X) and Target Vector (y)
        features = [c for c in df.columns if c not in [target_col, 'min', 'max']] # Exclude min/max as they are proxies for target
        X = df[features]
        y = df[target_col]

        # Train-Test Split (Time-based split, not random shuffle)
        test_size = int(len(df) * 0.2)
        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

        # Grid Search for Hyperparameters
        param_grid = {
            'n_estimators': [50, 100], # Number of trees
            'max_depth': [5, 10, None], # Max depth of trees
        }

        rf_model = RandomForestRegressor(random_state=42)
        # Cross-validation suitable for time series (respects order)
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error', # Optimize for MAE
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        predictions = best_rf.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        print(f"Best RF Parameters: {grid_search.best_params_}")
        print(f"Random Forest MAE: {mae:.2f}")
        
        return best_rf, mae

    def train_lstm(self, df, target_col='aqi', look_back=3):
        """
        Trains an LSTM (Long Short-Term Memory) network.
        """
        print("\n--- Training LSTM Model ---")
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        features = [c for c in df.columns if c not in ['min', 'max']]
        
        # Normalize data to [0, 1] range for LSTM stability
        data_values = df[features].values
        scaled_data = scaler.fit_transform(data_values)
        
        X, y = [], []
        target_idx = features.index(target_col)

        # Create Sequences (Rolling Window) for LSTM
        # Input: Sequence of 'look_back' days -> Output: Next day's AQI
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, :]) 
            y.append(scaled_data[i, target_idx]) 
            
        X, y = np.array(X), np.array(y)
        
        test_size = int(len(X) * 0.2)
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        # Build LSTM Architecture
        model = Sequential()
        model.add(LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))) # Layer 1
        model.add(Dropout(0.2)) # Regularization to prevent overfitting
        model.add(LSTM(16, return_sequences=False)) # Layer 2
        model.add(Dropout(0.2))
        model.add(Dense(1)) # Output Layer (Single value prediction)
        
        model.compile(optimizer='adam', loss='mae')
        model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test), verbose=1)
        
        predictions_scaled = model.predict(X_test)
        
        # Inverse transform
        dummy_matrix = np.zeros((len(predictions_scaled), len(features)))
        dummy_matrix[:, target_idx] = predictions_scaled.flatten()
        predictions = scaler.inverse_transform(dummy_matrix)[:, target_idx]
        
        dummy_matrix_y = np.zeros((len(y_test), len(features)))
        dummy_matrix_y[:, target_idx] = y_test
        actuals = scaler.inverse_transform(dummy_matrix_y)[:, target_idx]
        
        mae = mean_absolute_error(actuals, predictions)
        print(f"LSTM MAE: {mae:.2f}")
        return model, mae

    def generate_forecast(self, model, df, metadata, days=7):
        """
        Generates a 7-day forecast using the trained model recursively.
        Includes Location, Lat, Lon in the output.
        """
        print(f"\n--- Generating {days} Day Forecast ---")
        
        # Get the last available data point to start the recursive prediction
        last_data = df.iloc[-1].copy()
        last_date = df.index[-1]
        
        future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
        forecast_values = []
        
        # We need to maintain a moving window of past AQI values for lags to feed into prediction
        # [t-3, t-2, t-1]
        # In our df, we have columns lag_1d, lag_2d, lag_3d. 
        # But for recursion, it's easier to just track the actual past values in a list.
        past_aqi = [df.iloc[-i]['aqi'] for i in range(3, 0, -1)] # [lag_3, lag_2, lag_1]
        
        for future_date in future_dates:
            # Create a single-row dataframe for prediction for 'future_date'
            
            # 1. Calculate Date features for the forecast day
            day_of_week = future_date.dayofweek
            day_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            # 2. Retrieve Lag features from our 'past_aqi' tracker
            lag_1d = past_aqi[-1] # Most recent
            lag_2d = past_aqi[-2]
            lag_3d = past_aqi[-3]
            
            # Construct input vector matching the model's training features
            input_data = pd.DataFrame([{
                'day_of_week': day_of_week,
                'month': future_date.month,
                'day_sin': day_sin,
                'day_cos': day_cos,
                'lag_1d': lag_1d,
                'lag_2d': lag_2d,
                'lag_3d': lag_3d
            }])
            
            # Predict AQI for this future day
            pred_aqi = model.predict(input_data)[0]
            forecast_values.append(pred_aqi)
            
            # Update past_aqi list for the NEXT iteration (Recursive Forecasting)
            past_aqi.append(pred_aqi)
            past_aqi.pop(0) # Remove the oldest lag to slide the window forward
        
        # Create Result DataFrame
        forecast_df = pd.DataFrame({
            'Day': [f"Day-{i}" for i in range(1, days + 1)],
            'Date': future_dates,
            'Location': [metadata.get('city', 'Unknown')] * days,
            'Latitude': [metadata.get('lat', 0)] * days,
            'Longitude': [metadata.get('lon', 0)] * days,
            'Predicted AQI': [round(x, 2) for x in forecast_values],
            'Status': [self._get_aqi_status(x) for x in forecast_values]
        })
        return forecast_df

    def generate_hourly_forecast(self, current_aqi_pred):
        """
        Generates a simulated 24-hour forecast based on the predicted Daily Average.
        Uses a standard diurnal profile (traffic peaks in morn/eve) to distribute the AQI.
        """
        print("\n--- Generating 24-Hour Hourly Forecast (Simulated) ---")
        
        hours = list(range(24))
        hourly_aqi = []
        
        # Standard Diurnal Profile Factors (approximate traffic/pollution curve)
        # Represents typical daily fluctuation: Peaks around 9AM (Rush hour) and 8PM, Dip around 3AM (Night)
        # These factors are normalized so their mean defines the overall level.
        diurnal_profile = [
            0.8, 0.75, 0.7, 0.7, 0.75, 0.85, 1.0, 1.2, 1.3, 1.2, # 00-09
            1.1, 1.0, 0.9, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.25, # 10-19
            1.15, 1.0, 0.9, 0.85                                      # 20-23
        ]
        
        # Adjust profile key so that the average matches our predicted daily average AQI
        profile_mean = sum(diurnal_profile) / 24
        scaling_factor = current_aqi_pred / profile_mean
        
        current_hour_idx = datetime.now().hour
        
        future_hours = []
        for i in range(24):
            # Calculate next hours iteratively
            h_idx = (current_hour_idx + 1 + i) % 24
            time_str = (datetime.now() + timedelta(hours=i+1)).strftime("%Y-%m-%d %H:00")
            
            # Apply scaling factor to the profile
            predicted_h_aqi = diurnal_profile[h_idx] * scaling_factor
            
            future_hours.append({
                'Time': time_str,
                'Predicted AQI': round(predicted_h_aqi, 2),
                'Status': self._get_aqi_status(predicted_h_aqi)
            })
            
        return pd.DataFrame(future_hours)

    def _get_aqi_status(self, aqi):
        """
        Maps AQI numerical values to standard AQI health categories.
        """
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Moderate"
        elif aqi <= 150: return "Unhealthy for Sensitive Groups"
        elif aqi <= 200: return "Unhealthy"
        elif aqi <= 300: return "Very Unhealthy"
        else: return "Hazardous"

def detect_user_location():
    """
    Explicitly detects user location using IP-based geocoding.
    Returns lat, lon or None, None if failed.
    """
    if not GEOCODER_AVAILABLE:
        return None, None
        
    print("\n--- Step 1: Detecting User Location ---")
    try:
        g = geocoder.ip('me')
        if g.ok:
            print(f"Success! Detected Location: {g.city}, {g.country}")
            print(f"Coordinates: {g.lat}, {g.lng}")
            return g.lat, g.lng
        else:
            print("Could not detect location from IP. Will rely on API default.")
            return None, None
    except Exception as e:
        print(f"Location detection failed: {e}")
        return None, None

if __name__ == "__main__":
    print("Initializing AQI Prediction Pipeline...")
    
    # 1. Location Detection
    user_lat, user_lon = detect_user_location()
    
    # 2. Fetch Data (passing detected location)
    print("\n--- Step 2: Fetching Air Quality Data ---")
    fetcher = AQIDataFetcher(api_token=API_KEY, lat=user_lat, lon=user_lon)
    df, metadata = fetcher.fetch_data()
    
    # 3. Preprocessing & Augmentation
    print("\n--- Step 3: Preprocessing & Model Training ---")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess_and_augment(df)

    if not df_processed.empty:
        trainer = AQIModels()
        
        # Random Forest
        rf_model, rf_mae = trainer.train_random_forest(df_processed)
        
        if TF_AVAILABLE:
            # LSTM
            lstm_model, lstm_mae = trainer.train_lstm(df_processed)
        else:
            lstm_model, lstm_mae = None, float('inf')
        
        # 4. Generate Future Forecasts
        
        # A. 7-Day Forecast
        forecast_df = trainer.generate_forecast(rf_model, df_processed, metadata, days=7)
        
        # B. 24-Hour Forecast (Using Day-1 prediction as the baseline)
        day_1_pred = forecast_df.iloc[0]['Predicted AQI']
        hourly_df = trainer.generate_hourly_forecast(day_1_pred)
        
        print("\n=== FINAL RESULTS (Daily Average AQI) ===")
        print(f"Random Forest MAE: {rf_mae:.4f}")
        if TF_AVAILABLE:
            print(f"LSTM MAE:          {lstm_mae:.4f}")
        
        print("\n=== 24-HOUR HOURLY FORECAST ===")
        print(hourly_df.to_string(index=False))
        
        print("\n=== 7-DAY AQI FORECAST ===")
        print(forecast_df.to_string(index=False))
        print("==========================")

