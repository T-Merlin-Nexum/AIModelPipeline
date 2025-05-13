"""
Configuration settings for the ML Pipeline
"""

YOLO_MODEL_CONFIGS = {
    "yolov5s": {
        "default_epochs": 50,
        "default_batch_size": 16,
        "default_img_size": 640,
        "default_learning_rate": 0.01,
        "description": "YOLOv5 Small - Fastest model with good accuracy"
    },
    "yolov5m": {
        "default_epochs": 50,
        "default_batch_size": 16,
        "default_img_size": 640,
        "default_learning_rate": 0.01,
        "description": "YOLOv5 Medium - Balanced speed and accuracy"
    },
    "yolov5l": {
        "default_epochs": 50,
        "default_batch_size": 8,
        "default_img_size": 640,
        "default_learning_rate": 0.01,
        "description": "YOLOv5 Large - Higher accuracy, slower speed"
    },
    "yolov8n": {
        "default_epochs": 50,
        "default_batch_size": 16,
        "default_img_size": 640,
        "default_learning_rate": 0.01,
        "description": "YOLOv8 Nano - Fastest model"
    },
    "yolov8s": {
        "default_epochs": 50,
        "default_batch_size": 16,
        "default_img_size": 640,
        "default_learning_rate": 0.01,
        "description": "YOLOv8 Small - Fast with good accuracy"
    },
    "yolov8m": {
        "default_epochs": 50,
        "default_batch_size": 16,
        "default_img_size": 640,
        "default_learning_rate": 0.01,
        "description": "YOLOv8 Medium - Balanced performance"
    }
}

RF_DETR_MODEL_CONFIGS = {
    "rf_detr_r50": {
        "default_epochs": 50,
        "default_batch_size": 8,
        "default_img_size": 800,
        "default_learning_rate": 0.0001,
        "description": "RF-DETR with ResNet-50 backbone"
    },
    "rf_detr_r101": {
        "default_epochs": 50,
        "default_batch_size": 4,
        "default_img_size": 800,
        "default_learning_rate": 0.0001,
        "description": "RF-DETR with ResNet-101 backbone"
    }
}

# Dataset formats
SUPPORTED_ANNOTATION_FORMATS = [
    {
        "name": "COCO JSON",
        "description": "COCO dataset format with JSON annotations",
        "file_extensions": [".json"]
    },
    {
        "name": "YOLO TXT",
        "description": "YOLO format with one txt file per image",
        "file_extensions": [".txt"]
    },
    {
        "name": "Pascal VOC XML",
        "description": "Pascal VOC dataset format with XML annotations",
        "file_extensions": [".xml"]
    }
]

# Valid image file extensions
VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']

# MLFlow configuration
MLFLOW_EXPERIMENT_NAME = "object-detection-models"
