import sys
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Add project root to sys.path to import aqi_pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from aqi_pipeline import AQIModels, DisasterPredictor, AQIDataFetcher, WeatherFetcher, DataPreprocessor, TF_AVAILABLE
except ImportError as e:
    print(f"Error importing aqi_pipeline: {e}")
    # Mock classes to prevent crash if pipeline is missing during config
    class AQIModels: pass
    class DisasterPredictor: pass

from app.models import db, APIEnvironmentalData, AQIPredictions, DisasterPredictions

# Global model cache
MODEL_CACHE = {
    'rf': None,
    'lstm': None,
    'disaster': None
}

class MLService:
    def __init__(self):
        self.aqi_trainer = AQIModels()
        self.disaster_predictor = DisasterPredictions() # This is the DB model, we need the Logic class
        self.disaster_logic = DisasterPredictor()
        self.weather_fetcher = WeatherFetcher()
        self.preprocessor = DataPreprocessor()
        self.models_dir = os.path.join(os.path.dirname(__file__), '../ml_models')
        
        # Ensure models dir exists
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load_or_train_models(self):
        """
        Loads models from disk or triggers training if not found.
        """
        rf_path = os.path.join(self.models_dir, 'rf_model.pkl')
        disaster_path = os.path.join(self.models_dir, 'disaster_model.pkl')
        
        # Load RF Model
        if os.path.exists(rf_path):
            MODEL_CACHE['rf'] = joblib.load(rf_path)
            print("Loaded Random Forest model from disk.")
        else:
            print("RF model not found. Retraining...")
            self._train_and_save_all()

        # Load Disaster Model
        if os.path.exists(disaster_path):
            MODEL_CACHE['disaster'] = joblib.load(disaster_path)
             # Re-attach model to the class instance
            self.disaster_logic.model = MODEL_CACHE['disaster']
            print("Loaded Disaster model from disk.")
        else:
             if not MODEL_CACHE['rf']: # If we didn't train above
                 self._train_and_save_all()

    def _train_and_save_all(self):
        """
        Trains all models using the pipeline and saves them.
        """
        print("Training models... This may take a moment.")
        # 1. Fetch some data for training (Auto location or default)
        fetcher = AQIDataFetcher(api_token="24da0fb52646fa4b9ad08d97d043ea86d2c31983") 
        df, _, _ = fetcher.fetch_data()
        
        if df.empty:
            print("Failed to fetch data for training.")
            return

        # 2. Preprocess
        df_processed = self.preprocessor.preprocess_and_augment(df)
        
        # 3. Train RF
        rf_model, _ = self.aqi_trainer.train_random_forest(df_processed)
        MODEL_CACHE['rf'] = rf_model
        joblib.dump(rf_model, os.path.join(self.models_dir, 'rf_model.pkl'))
        
        # 4. Train Disaster
        from aqi_pipeline import DisasterDataGenerator
        gen = DisasterDataGenerator()
        df_disaster = gen.generate_synthetic_data()
        disaster_model = self.disaster_logic.train_model(df_disaster)
        MODEL_CACHE['disaster'] = disaster_model
        # We need to save the underlying sklearn model
        joblib.dump(disaster_model, os.path.join(self.models_dir, 'disaster_model.pkl'))
        
        print("All models trained and saved.")

    def predict_aqi(self, lat, lon):
        """
        1. Fetches real data for location.
        2. Predicts 7-day forecast.
        3. Saves to DB.
        4. Returns JSON.
        """
        # Ensure models are loaded
        if not MODEL_CACHE['rf']:
            self.load_or_train_models()
            
        # Fetch Live Data
        fetcher = AQIDataFetcher(api_token="24da0fb52646fa4b9ad08d97d043ea86d2c31983", lat=lat, lon=lon)
        df, metadata, weather_data = fetcher.fetch_data()
        
        # Save Current Data to DB
        self._save_environmental_data(metadata, weather_data, lat, lon)

        if df.empty:
            return {"error": "Could not fetch data for location"}

        # Preprocess
        df_processed = self.preprocessor.preprocess_and_augment(df)
        
        # Generate Forecast
        forecast_df = self.aqi_trainer.generate_forecast(
            MODEL_CACHE['rf'], 
            df_processed, 
            metadata, 
            self.disaster_logic, 
            self.weather_fetcher, 
            days=7
        )
        
        # Save Predictions to DB
        saved_predictions = self._save_predictions(forecast_df)
        
        return {
            "location": metadata,
            "current_weather": weather_data,
            "forecast": forecast_df.to_dict(orient='records')
        }

    def _save_environmental_data(self, metadata, weather, lat, lon):
        new_entry = APIEnvironmentalData(
            latitude=lat,
            longitude=lon,
            aqi=metadata.get('current_aqi'),
            pm25=weather.get('pm25'),
            pm10=weather.get('pm10'),
            temperature=weather.get('temp'),
            humidity=weather.get('humidity'),
            pressure=weather.get('pressure'),
            wind_speed=weather.get('wind_speed'),
            data_source='WAQI_OpenMeteo'
        )
        db.session.add(new_entry)
        db.session.commit()

    def _save_predictions(self, forecast_df):
        saved = []
        for _, row in forecast_df.iterrows():
            # AQI Prediction
            aqi_pred = AQIPredictions(
                prediction_date=row['Date'],
                predicted_aqi=row['Predicted AQI'],
                model_name='RandomForest',
                model_accuracy=0.85 # Placeholder or pass real accuracy
            )
            db.session.add(aqi_pred)
            
            # Disaster Risk
            # Parse "Flood (80%)" string logic or update pipeline to return easier format
            # For now, simplistic parsing
            risk_str = row['Disaster Risk']
            is_disaster = risk_str != "None"
            risk_type = risk_str.split('(')[0].strip() if is_disaster else "None"
            
            disaster_pred = DisasterPredictions(
                prediction_date=row['Date'],
                disaster_occurred=is_disaster,
                disaster_type=risk_type,
                risk_level=row['Status'], # Using AQI status as proxy for risk level for now
                model_name='DisasterMatrix'
            )
            db.session.add(disaster_pred)
            saved.append(aqi_pred)
            
        db.session.commit()
        return saved
