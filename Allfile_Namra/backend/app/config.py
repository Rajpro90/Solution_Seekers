import os

class Config:
    # Use SQLite for easy local setup, can be swapped for MySQL/PostgreSQL
    # URI format for MySQL: mysql+pymysql://user:password@localhost/disaster_aqi_system
    basedir = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, '../../disaster_db.sqlite')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-for-local-testing'
