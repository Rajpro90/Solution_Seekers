from flask import Flask
import os

from flask_cors import CORS
from app.config import Config
from app.extensions import db

def create_app(config_class=Config):
    # Point to project root for templates and static files to serve existing frontend
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    app = Flask(__name__, template_folder=project_root, static_folder=project_root, static_url_path='')

    app.config.from_object(config_class)

    # Database configuration
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    db_path = os.path.join(basedir, 'disaster_db.sqlite')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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
