import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mpasas-secret-2025-camr')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
    TEMPLATE_IMAGES = os.path.join(BASE_DIR, 'uploads', 'templates')
    SCRIPT_IMAGES  = os.path.join(BASE_DIR, 'uploads', 'scripts')
    CHARTS_FOLDER  = os.path.join(BASE_DIR, 'static', 'charts')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024   # 100 MB
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'mpasas.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}
