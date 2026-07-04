import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.environ.get('MPASAS_DATA_DIR', os.path.join(BASE_DIR, 'instance'))
UPLOAD_DIR = os.environ.get('MPASAS_UPLOAD_DIR', os.path.join(BASE_DIR, 'uploads'))
RESULTS_DIR = os.environ.get('MPASAS_RESULTS_DIR', os.path.join(BASE_DIR, 'results'))
DB_PATH = os.environ.get('MPASAS_DB_PATH', os.path.join(DATA_DIR, 'mpasas.db'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mpasas-secret-2025-camr')
    UPLOAD_FOLDER = UPLOAD_DIR
    RESULTS_FOLDER = RESULTS_DIR
    TEMPLATE_IMAGES = os.path.join(UPLOAD_DIR, 'templates')
    SCRIPT_IMAGES  = os.path.join(UPLOAD_DIR, 'scripts')
    CHARTS_FOLDER  = os.path.join(BASE_DIR, 'static', 'charts')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024   # 100 MB
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}
