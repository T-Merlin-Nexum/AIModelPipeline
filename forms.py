from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, PasswordField, SubmitField, TextAreaField, SelectField, RadioField
from wtforms import IntegerField, FloatField, BooleanField, HiddenField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError, NumberRange, Optional
from config import YOLO_MODEL_CONFIGS, RF_DETR_MODEL_CONFIGS


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')


class DatasetUploadForm(FlaskForm):
    dataset_name = StringField('Dataset Name', validators=[DataRequired()])
    description = TextAreaField('Description')
    dataset_zip = FileField('Dataset ZIP', validators=[DataRequired(), FileAllowed(['zip'])])
    format_type = RadioField('Format Type', choices=[
        ('yolo', 'YOLO TXT'),
        ('coco', 'COCO JSON'),
        ('voc', 'Pascal VOC')
    ], default='yolo', validators=[DataRequired()])
    submit = SubmitField('Upload Dataset')


class YOLOConfigForm(FlaskForm):
    job_name = StringField('Job Name', validators=[DataRequired(), Length(min=3, max=100)])
    dataset_id = SelectField('Dataset', coerce=int, validators=[DataRequired()])
    model_variant = SelectField('Model Variant', choices=[
        ('yolov5s', 'YOLOv5 Small'),
        ('yolov5m', 'YOLOv5 Medium'),
        ('yolov5l', 'YOLOv5 Large'),
        ('yolov8n', 'YOLOv8 Nano'),
        ('yolov8s', 'YOLOv8 Small'),
        ('yolov8m', 'YOLOv8 Medium')
    ], validators=[DataRequired()])
    epochs = IntegerField('Epochs', validators=[DataRequired(), NumberRange(min=1, max=500)])
    batch_size = IntegerField('Batch Size', validators=[DataRequired(), NumberRange(min=1, max=64)])
    img_size = IntegerField('Image Size', validators=[DataRequired(), NumberRange(min=320, max=1280)])
    learning_rate = FloatField('Learning Rate', validators=[DataRequired(), NumberRange(min=0.00001, max=0.1)])
    pretrained = BooleanField('Use Pretrained Weights', default=True)
    submit = SubmitField('Start Training')


class RFDETRConfigForm(FlaskForm):
    job_name = StringField('Job Name', validators=[DataRequired(), Length(min=3, max=100)])
    dataset_id = SelectField('Dataset', coerce=int, validators=[DataRequired()])
    model_variant = SelectField('Model Variant', choices=[
        ('rf_detr_r50', 'RF-DETR ResNet-50'),
        ('rf_detr_r101', 'RF-DETR ResNet-101')
    ], validators=[DataRequired()])
    epochs = IntegerField('Epochs', validators=[DataRequired(), NumberRange(min=1, max=300)])
    batch_size = IntegerField('Batch Size', validators=[DataRequired(), NumberRange(min=1, max=32)])
    img_size = IntegerField('Image Size', validators=[DataRequired(), NumberRange(min=400, max=1333)])
    learning_rate = FloatField('Learning Rate', validators=[DataRequired(), NumberRange(min=0.000001, max=0.01)])
    pretrained = BooleanField('Use Pretrained Weights', default=True)
    submit = SubmitField('Start Training')