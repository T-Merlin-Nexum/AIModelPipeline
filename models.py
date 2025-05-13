from app import db
from flask_login import UserMixin
from datetime import datetime
import json
from werkzeug.security import generate_password_hash, check_password_hash


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    datasets = db.relationship('Dataset', backref='owner', lazy='dynamic')
    training_jobs = db.relationship('TrainingJob', backref='owner', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text)
    data_path = db.Column(db.String(256), nullable=False)
    format_type = db.Column(db.String(20), nullable=False)  # COCO, YOLO, VOC, etc.
    class_names = db.Column(db.Text)  # JSON-serialized list of class names
    image_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    training_jobs = db.relationship('TrainingJob', backref='dataset', lazy='dynamic')
    
    def get_class_names(self):
        if self.class_names:
            return json.loads(self.class_names)
        return []
    
    def set_class_names(self, class_list):
        self.class_names = json.dumps(class_list)
    
    def __repr__(self):
        return f'<Dataset {self.name}>'


class TrainingJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_name = db.Column(db.String(120), nullable=False)
    model_type = db.Column(db.String(20), nullable=False)  # YOLO, RF-DETR
    model_variant = db.Column(db.String(20), nullable=False)  # e.g., yolov5s, rf_detr_r50
    hyperparameters = db.Column(db.Text, nullable=False)  # JSON-serialized hyperparameters
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed
    mlflow_run_id = db.Column(db.String(36))  # MLFlow run ID
    run_id = db.Column(db.String(36))  # Run ID for tracking
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    error_message = db.Column(db.Text)
    
    # Foreign Keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    
    def get_hyperparameters(self):
        return json.loads(self.hyperparameters)
    
    def set_hyperparameters(self, params_dict):
        self.hyperparameters = json.dumps(params_dict)
    
    def __repr__(self):
        return f'<TrainingJob {self.job_name}>'


class ModelArtifact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    training_job_id = db.Column(db.Integer, db.ForeignKey('training_job.id'), nullable=False)
    artifact_path = db.Column(db.String(256), nullable=False)
    artifact_type = db.Column(db.String(20), nullable=False)  # weights, config, metrics
    metrics = db.Column(db.Text)  # JSON-serialized metrics
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    training_job = db.relationship('TrainingJob', backref='artifacts')
    
    def get_metrics(self):
        if self.metrics:
            return json.loads(self.metrics)
        return {}
    
    def set_metrics(self, metrics_dict):
        self.metrics = json.dumps(metrics_dict)
    
    def __repr__(self):
        return f'<ModelArtifact {self.artifact_type} for job {self.training_job_id}>'
