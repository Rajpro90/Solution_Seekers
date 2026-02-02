from flask import Flask
from flask_cors import CORS
from app.config import Config
from app.extensions import db

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    CORS(app) # Enable CORS for frontend communication

    # Register blueprints (we will create routes next)
    from app.routes import main
    app.register_blueprint(main)

    # Create tables if they don't exist
    with app.app_context():
        db.create_all()

    return app
