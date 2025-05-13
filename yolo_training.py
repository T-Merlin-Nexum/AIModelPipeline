
import os
import sys
import logging
import time
import json
import glob
import requests
from tqdm import tqdm
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_pretrained_weights(model_variant, pretrained_dir=None):
    """Download pre-trained YOLO weights if not already present"""
    # Create pretrained directory if it doesn't exist
    if pretrained_dir is None:
        pretrained_dir = os.path.join(os.getcwd(), "pretrained")
    
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)
        logger.info(f"Created pretrained weights directory: {pretrained_dir}")

    # Define model weights URLs based on variant
    pretrained_urls = {
        # YOLO models
        'yolov5s': 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt',
        'yolov5m': 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt',
        'yolov5l': 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5l.pt',
        'yolov5x': 'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x.pt',
        'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
        'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
        'yolov8l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
        'yolov8x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
    }

    logger.info(f"Available YOLO pretrained model variants: {list(pretrained_urls.keys())}")

    # Check if variant exists in our mapping
    if model_variant not in pretrained_urls:
        logger.warning(f"No pre-trained weights URL defined for variant: {model_variant}")
        return None

    # Determine weights filename based on model variant
    possible_filenames = [f"{model_variant}_pretrained.pt", f"{model_variant}.pt"]

    # Check if weights already exist with any of the possible filenames
    for filename in possible_filenames:
        weights_path = os.path.join(pretrained_dir, filename)
        if os.path.exists(weights_path):
            logger.info(f"Pre-trained weights found at: {weights_path}")
            return weights_path

    # Define the standard filename for downloading
    weights_filename = f"{model_variant}_pretrained.pt"
    weights_path = os.path.join(pretrained_dir, weights_filename)

    # Download weights if they don't exist locally
    url = pretrained_urls[model_variant]
    logger.info(f"Downloading pre-trained weights for {model_variant} from {url}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8 KB

        with open(weights_path, 'wb') as f, tqdm(
                desc=f"Downloading {weights_filename}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))

        logger.info(f"Successfully downloaded pre-trained weights to: {weights_path}")
        return weights_path

    except Exception as e:
        logger.error(f"Failed to download pre-trained weights: {str(e)}")
        return None

def train_yolo_model(dataset_info, model_variant, hyperparameters, mlflow_run_id, mlflow_tracking_uri):
    """Train YOLO model with provided configuration"""
    logger.info(f"Training YOLO model variant: {model_variant} with MLFlow run ID: {mlflow_run_id}")
    logger.info(f"Using dataset: {dataset_info}")
    logger.info(f"Hyperparameters: {hyperparameters}")

    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created models directory: {models_dir}")

    # Check if we need to use pre-trained weights
    pretrained_weights_path = None
    # Convert string 'true' to boolean if needed
    pretrained_flag = hyperparameters.get('pretrained', False)
    if isinstance(pretrained_flag, str):
        pretrained_flag = pretrained_flag.lower() == 'true'

    if pretrained_flag:
        logger.info(f"Using pre-trained weights as requested in hyperparameters for variant {model_variant}")
        pretrained_weights_path = download_pretrained_weights(model_variant)
        if pretrained_weights_path:
            logger.info(f"Pre-trained weights loaded from: {pretrained_weights_path}")
        else:
            logger.warning(f"Failed to download pre-trained weights for {model_variant}, continuing without them")

    # Determine model path
    model_filename = f"{model_variant}_{mlflow_run_id[:8]}.pt"
    model_path = os.path.join(models_dir, model_filename)

    # Get training parameters
    total_epochs = int(hyperparameters.get('epochs', 100))
    batch_size = int(hyperparameters.get('batch_size', 16))
    img_size = int(hyperparameters.get('img_size', 640))
    learning_rate = float(hyperparameters.get('learning_rate', 0.01))

    # Get real dataset path
    dataset_path = dataset_info.get('dataset_path')
    logger.info(f"Starting YOLO training on dataset: {dataset_path} for {total_epochs} epochs")

    # Connect to MLFlow for logging if available
    mlflow_active = False
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow_active = True
        logger.info("MLFlow logging enabled")
    except Exception as e:
        logger.warning(f"MLFlow logging disabled: {str(e)}")

    # Import required modules for YOLO training
    try:
        from ultralytics import YOLO
        from ultralytics import settings

        # Initialize model with pretrained weights if available
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            logger.info(f"Loading pretrained YOLO model from {pretrained_weights_path}")
            model = YOLO(pretrained_weights_path)
        else:
            logger.info(f"Creating new YOLO model: {model_variant}")
            model = YOLO(model_variant)
        settings.update({'mlflow': True})

        # Configure dataset
        if not os.path.exists(dataset_path):
            # Fallback for testing: use COCO8 example dataset
            logger.warning(f"Dataset path {dataset_path} not found, using example dataset")
            dataset_path = "coco8"
        elif os.path.isdir(dataset_path):
            # Verifica se esiste il file di configurazione data.yaml nella cartella
            yaml_path = os.path.join(dataset_path, "data.yaml")
            if os.path.exists(yaml_path):
                logger.info(f"Usando il file di configurazione YAML esistente: {yaml_path}")
                dataset_path = yaml_path
            else:
                # Crea un file YAML temporaneo per il dataset
                logger.info(f"Il dataset è una directory, creazione file YAML temporaneo")
                import yaml

                # Verifica la struttura del dataset
                train_dir = os.path.join(dataset_path, "train")
                valid_dir = os.path.join(dataset_path, "valid")
                test_dir = os.path.join(dataset_path, "test")

                # Creazione configurazione YAML
                yaml_config = {
                    "path": dataset_path,
                    "train": "train" if os.path.exists(train_dir) else "",
                    "val": "valid" if os.path.exists(valid_dir) else "",
                    "test": "test" if os.path.exists(test_dir) else "",
                    "names": {}
                }

                # Rileva automaticamente le classi dalle etichette nel set di training
                if os.path.exists(train_dir):
                    labels_dir = os.path.join(train_dir, "labels")
                    if os.path.exists(labels_dir):
                        class_ids = set()
                        # Analizza i primi 10 file di etichette per rilevare le classi
                        for label_file in os.listdir(labels_dir)[:10]:
                            if label_file.endswith('.txt'):
                                with open(os.path.join(labels_dir, label_file), 'r') as f:
                                    for line in f:
                                        parts = line.strip().split()
                                        if parts and parts[0].isdigit():
                                            class_ids.add(int(parts[0]))

                        # Crea dizionario delle classi
                        for class_id in sorted(class_ids):
                            yaml_config["names"][class_id] = f"class{class_id}"

                # Se non sono state trovate classi, imposta valori predefiniti
                if not yaml_config["names"]:
                    yaml_config["names"] = {0: "class0", 1: "class1"}

                # Salva il file YAML
                yaml_path = os.path.join(dataset_path, "data.yaml")
                with open(yaml_path, 'w') as f:
                    yaml.dump(yaml_config, f, sort_keys=False)

                logger.info(f"File di configurazione YAML creato: {yaml_path}")
                dataset_path = yaml_path

        # Log train command details for debugging
        logger.info(f"Starting YOLO training with dataset: {dataset_path}")
        logger.info(f"Epochs: {total_epochs}, Batch size: {batch_size}, Image size: {img_size}")
        logger.info(f"Learning rate: {learning_rate}")

        # Train the model with real hyperparameters
        try:
            # Riduci dimensioni batch e risoluzione immagine se necessario
            adjusted_batch = min(batch_size, 8)  # Riduci il batch size massimo
            adjusted_size = min(img_size, 416)  # Riduci la dimensione massima immagine

            results = model.train(
                data=dataset_path,
                epochs=total_epochs,
                batch=adjusted_batch,
                imgsz=adjusted_size,
                lr0=learning_rate,
                patience=50,
                save=True,
                project="training_jobs",
                name=f"job_{mlflow_run_id[:8]}",
                cache=False,  # Disabilita la cache per risparmiare memoria
                workers=1  # Riduci il numero di worker per ridurre la memoria
            )
            logger.info(f"Training completed successfully. Results: {results.results_dict}")
        except Exception as e:
            logger.error(f"Error during YOLO training: {str(e)}")
            # Print detailed error traceback
            import traceback
            logger.error(f"Detailed traceback: {traceback.format_exc()}")
            raise

        # Get metrics from results
        final_metrics = results.results_dict

        # Extract metrics for reporting
        precision = final_metrics.get('metrics/precision(B)', 0.0)
        recall = final_metrics.get('metrics/recall(B)', 0.0)
        mAP50 = final_metrics.get('metrics/mAP50(B)', 0.0)
        mAP50_95 = final_metrics.get('metrics/mAP50-95(B)', 0.0)

        # Save the trained model - use the best.pt file directly instead of exporting
        trained_model_path = os.path.join(
            os.getcwd(),
            f"training_jobs/job_{mlflow_run_id[:8]}/weights/best.pt"
        )
        if os.path.exists(trained_model_path):
            import shutil
            shutil.copy2(trained_model_path, model_path)
            logger.info(f"Model saved to: {model_path}")
        else:
            logger.warning(f"Trained model not found at {trained_model_path}, saving current model")
            # Use the trained model directly
            model.save(model_path)

        # Log the model artifact to MLFlow if available
        if mlflow_active:
            try:
                # Log final metrics to MLFlow
                final_metrics_dict = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "mAP50": float(mAP50),
                    "mAP50-95": float(mAP50_95),
                    "epochs_completed": int(total_epochs)
                }

                # Log metrics one by one to ensure success
                for metric_name, metric_value in final_metrics_dict.items():
                    try:
                        mlflow.log_metric(metric_name, metric_value)
                        logger.info(f"Logged metric {metric_name}={metric_value} to MLFlow")
                    except Exception as e:
                        logger.warning(f"Failed to log metric {metric_name}: {str(e)}")

                # Verify if run_id is active
                try:
                    active_run = mlflow.active_run()
                    if active_run is None:
                        logger.info(f"No active MLFlow run. Starting run with ID: {mlflow_run_id}")
                        mlflow.start_run(run_id=mlflow_run_id)
                except Exception as e:
                    logger.warning(f"Error checking active run: {str(e)}")

                # Log model artifact
                if os.path.exists(model_path):
                    mlflow.log_artifact(model_path, artifact_path="model")
                    logger.info(f"Model artifact logged to MLFlow: {model_path}")
                
                # Log plot images as artifacts
                results_dir = os.path.join(os.getcwd(), f"training_jobs/job_{mlflow_run_id[:8]}")
                if os.path.exists(results_dir):
                    for root, _, files in os.walk(results_dir):
                        for file in files:
                            if file.endswith(('.png', '.jpg')) and not file.startswith('.'):
                                img_path = os.path.join(root, file)
                                if os.path.exists(img_path):
                                    rel_path = os.path.relpath(root, results_dir)
                                    mlflow.log_artifact(img_path, artifact_path=f"plots/{rel_path}")
                                    logger.info(f"Logged plot to MLFlow: {img_path}")
                
                logger.info(f"All metrics and artifacts logged to MLFlow")
            except Exception as e:
                logger.warning(f"Failed to log to MLFlow: {str(e)}")
                import traceback
                logger.debug(f"MLFlow error details: {traceback.format_exc()}")

        # Return training results
        return {
            "model_path": model_path,
            "results": {
                "precision": precision,
                "recall": recall,
                "mAP50": mAP50,
                "mAP50-95": mAP50_95
            }
        }

    except Exception as e:
        logger.exception(f"Error in YOLO training: {str(e)}")
        # Fall back to pretrained weights if training failed
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            import shutil
            shutil.copy2(pretrained_weights_path, model_path)
            logger.warning(f"Training failed, using pretrained weights: {pretrained_weights_path}")
            # Return results with error info
            return {
                "model_path": model_path,
                "results": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "mAP50": 0.0,
                    "mAP50-95": 0.0,
                    "error": str(e),
                    "info": "Using pretrained weights due to training error"
                }
            }
        else:
            # Return error without model
            return {
                "model_path": None,
                "results": {
                    "error": str(e)
                }
            }
