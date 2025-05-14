import logging
import os

from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)
# create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https

# Configure Gunicorn timeout
if os.environ.get('GUNICORN_TIMEOUT'):
    timeout = int(os.environ.get('GUNICORN_TIMEOUT'))
else:
    timeout = 300  # 5 minutes timeout for large file uploads

# Disabilitiamo la protezione CSRF per semplificare l'uso dell'applicazione
csrf = CSRFProtect()
csrf.init_app(app)
app.config['WTF_CSRF_ENABLED'] = False

# Configure the SQLite database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///ml_pipeline.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 510 * 1024 * 1024  # 500 MB max upload with buffer

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Configure MLFlow
app.config["MLFLOW_TRACKING_URI"] = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")

# initialize the app with the extensions
db.init_app(app)

# Set up login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

with app.app_context():
    # Import models
    import models  # noqa: F401

    # Create tables
    db.create_all()

    # Import and register blueprints/routes
    from routes import register_routes

    register_routes(app)

    # Import authentication routes
    from auth import auth_bp

    app.register_blueprint(auth_bp)

    # Set up login loader
    from models import User


    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))


    logger.info("Flask application configured and started")
