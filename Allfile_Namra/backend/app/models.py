from datetime import datetime
from app.extensions import db

class APIEnvironmentalData(db.Model):
    __tablename__ = 'api_environmental_data'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    aqi = db.Column(db.Integer)
    pm25 = db.Column(db.Float)
    pm10 = db.Column(db.Float)
    temperature = db.Column(db.Float)
    humidity = db.Column(db.Float)
    pressure = db.Column(db.Float)
    wind_speed = db.Column(db.Float)
    rainfall = db.Column(db.Float)
    data_source = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'location': {'lat': self.latitude, 'lon': self.longitude},
            'aqi': self.aqi,
            'metrics': {
                'pm25': self.pm25,
                'pm10': self.pm10,
                'temp': self.temperature,
                'humidity': self.humidity
            },
            'source': self.data_source
        }

class AQIPredictions(db.Model):
    __tablename__ = 'aqi_predictions'

    id = db.Column(db.Integer, primary_key=True)
    prediction_date = db.Column(db.DateTime, nullable=False)
    predicted_aqi = db.Column(db.Float)
    predicted_pm25 = db.Column(db.Float)
    predicted_pm10 = db.Column(db.Float)
    model_name = db.Column(db.String(100))
    model_accuracy = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'date': self.prediction_date.isoformat(),
            'predicted_aqi': self.predicted_aqi,
            'components': {
                'pm25': self.predicted_pm25,
                'pm10': self.predicted_pm10
            },
            'model': {
                'name': self.model_name,
                'accuracy': self.model_accuracy
            }
        }

class DisasterPredictions(db.Model):
    __tablename__ = 'disaster_predictions'

    id = db.Column(db.Integer, primary_key=True)
    prediction_date = db.Column(db.DateTime, nullable=False)
    disaster_occurred = db.Column(db.Boolean)
    disaster_probability = db.Column(db.Float)
    disaster_type = db.Column(db.String(100))
    model_name = db.Column(db.String(100))
    model_accuracy = db.Column(db.Float)
    risk_level = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'date': self.prediction_date.isoformat(),
            'prediction': {
                'is_disaster': self.disaster_occurred,
                'type': self.disaster_type,
                'probability': self.disaster_probability,
                'risk_level': self.risk_level
            }
        }
