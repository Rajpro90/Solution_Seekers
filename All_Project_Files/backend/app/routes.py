from flask import Blueprint, request, jsonify
from app.services import MLService
from app.models import APIEnvironmentalData, AQIPredictions
from app.extensions import db

main = Blueprint('main', __name__)
ml_service = MLService()

@main.route('/api/predict/aqi', methods=['POST'])
def predict_aqi():
    """
    Expects JSON: { "latitude": 12.34, "longitude": 56.78 }
    """
    data = request.get_json()
    lat = data.get('latitude')
    lon = data.get('longitude')
    
    if not lat or not lon:
        return jsonify({"error": "Latitude and Longitude are required"}), 400
        
    try:
        # Trigger the full pipeline
        result = ml_service.predict_aqi(lat, lon)
        return jsonify(result), 200
    except Exception as e:
        print(f"Error in predict_aqi: {e}")
        return jsonify({"error": str(e)}), 500

@main.route('/api/history', methods=['GET'])
def get_history():
    """
    Fetches the last 50 recorded environmental data points.
    """
    history = APIEnvironmentalData.query.order_by(APIEnvironmentalData.timestamp.desc()).limit(50).all()
    return jsonify([h.to_dict() for h in history]), 200

@main.route('/api/latest', methods=['GET'])
def get_latest():
    """
    Fetches the most recent prediction.
    """
    latest = AQIPredictions.query.order_by(AQIPredictions.created_at.desc()).first()
    if latest:
        return jsonify(latest.to_dict()), 200
    return jsonify({"message": "No data available"}), 404

@main.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "Backend is running", "service": "Disaster & AQI Prediction System"}), 200
