import logging
import os
import threading
import uuid

import yaml

from rfdetr_training import train_rfdetr_model, get_pretrained_model_info
from yolo_training import train_yolo_model, download_pretrained_weights as download_yolo_weights

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DirectTrainingPipeline:
    """Direct training pipeline implementation that replaces Dagster"""

    def __init__(self, app=None):
        self.app = app

    def load_dataset(self, job_id):
        """Load real dataset for training"""
        logger.info(f"Loading real dataset for job ID: {job_id}")
        try:
            # Import application models
            from models import TrainingJob, Dataset
            import os
            from app import db, app

            # Use app context
            with app.app_context():
                # Get job info
                job = db.session.get(TrainingJob, job_id)
                if not job:
                    logger.error(f"Job {job_id} not found")
                    return {"dataset_path": f"coco8", "format_type": "yolo"}

                # Get dataset info
                dataset = db.session.get(Dataset, job.dataset_id)
                if not dataset:
                    logger.error(f"Dataset {job.dataset_id} not found")
                    return {"dataset_path": f"coco8", "format_type": "yolo"}

                # Create yaml dataset file if needed
                dataset_path = dataset.data_path

                # Correggi il percorso per ambiente containerizzato
                if '/home/hb/lab/AIModelPipeline/' in dataset_path:
                    # Sostituisci il percorso assoluto con quello relativo a /app
                    dataset_path = dataset_path.replace(
                        '/home/hb/lab/AIModelPipeline/', '/app/')
                    logger.info(
                        f"Convertito percorso dataset per ambiente container: {dataset_path}"
                    )

                if not os.path.exists(dataset_path):
                    logger.warning(f"Dataset path {dataset_path} not found")
                    # Prova un percorso alternativo (in caso di montaggio diverso)
                    alt_path = os.path.join('/app', 'uploads',
                                            os.path.basename(dataset_path))
                    if os.path.exists(alt_path):
                        logger.info(
                            f"Dataset trovato al percorso alternativo: {alt_path}"
                        )
                        dataset_path = alt_path
                    else:
                        # Fallback to example dataset
                        return {
                            "dataset_path": f"coco8",
                            "format_type": "yolo"
                        }

                # Verifica se è un dataset YOLO e genera/aggiorna data.yaml se necessario
                if dataset.format_type == 'yolo':
                    yaml_path = os.path.join(dataset_path, 'data.yaml')
                    # Se esiste già, aggiorniamo solo il path assoluto
                    if os.path.exists(yaml_path):
                        logger.info(
                            f"Aggiornamento percorso in data.yaml esistente: {yaml_path}"
                        )
                        try:
                            with open(yaml_path, 'r') as f:
                                yaml_data = yaml.safe_load(f)
                            # Aggiorna il percorso
                            yaml_data['path'] = dataset_path
                            with open(yaml_path, 'w') as f:
                                yaml.dump(yaml_data,
                                          f,
                                          default_flow_style=False)
                        except Exception as e:
                            logger.warning(
                                f"Errore nell'aggiornamento del file data.yaml: {str(e)}"
                            )
                    else:
                        # Crea un nuovo file data.yaml
                        logger.info(
                            f"Creazione nuovo file data.yaml in {yaml_path}")
                        train_dir = os.path.join(dataset_path, 'train')
                        valid_dir = os.path.join(dataset_path, 'valid')
                        test_dir = os.path.join(dataset_path, 'test')

                        yaml_config = {
                            "path": dataset_path,
                            "train":
                                "train" if os.path.exists(train_dir) else "",
                            "val":
                                "valid" if os.path.exists(valid_dir) else "",
                            "test": "test" if os.path.exists(test_dir) else "",
                            "names": {
                                0: "class0",
                                1: "class1",
                                2: "class2"
                            }
                        }

                        # Salva il file YAML
                        with open(yaml_path, 'w') as f:
                            yaml.dump(yaml_config, f, default_flow_style=False)

                logger.info(
                    f"Found dataset at {dataset_path} with format {dataset.format_type}"
                )
                return {
                    "dataset_path": dataset_path,
                    "format_type": dataset.format_type
                }

                logger.info(
                    f"Found dataset at {dataset_path} with format {dataset.format_type}"
                )
                return {
                    "dataset_path": dataset_path,
                    "format_type": dataset.format_type
                }
        except Exception as e:
            logger.exception(f"Error loading dataset: {str(e)}")
            # Return default path when exceptions occur
            return {"dataset_path": f"coco8", "format_type": "yolo"}

    def download_pretrained_weights(self, model_variant):
        """Download pre-trained weights if not already present"""
        # Delegate to appropriate module based on model variant
        if 'yolo' in model_variant:
            return download_yolo_weights(model_variant)
        elif 'rf_detr' in model_variant:
            # RF-DETR models are already pretrained, just get model info
            from rfdetr_training import get_pretrained_model_info
            return get_pretrained_model_info(model_variant)
        else:
            logger.warning(f"Unknown model variant: {model_variant}, cannot determine weights")
            return None

    def train_model(self, dataset_info, model_variant, hyperparameters,
                    mlflow_run_id, mlflow_tracking_uri):
        """Train model with provided configuration using actual implementation"""
        logger.info(
            f"Training model variant: {model_variant} with MLFlow run ID: {mlflow_run_id}"
        )
        logger.info(f"Using dataset: {dataset_info}")
        logger.info(f"Hyperparameters: {hyperparameters}")

        # Create models directory if it doesn't exist
        import os
        models_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logger.info(f"Created models directory: {models_dir}")

        # Delegate to the appropriate training function based on model type
        if 'yolo' in model_variant:
            return train_yolo_model(
                dataset_info,
                model_variant,
                hyperparameters,
                mlflow_run_id,
                mlflow_tracking_uri
            )
        elif 'rf_detr' in model_variant:
            return train_rfdetr_model(
                dataset_info,
                model_variant,
                hyperparameters,
                mlflow_run_id,
                mlflow_tracking_uri
            )
        else:
            logger.error(f"Unsupported model variant: {model_variant}")
            return {
                "model_path": None,
                "results": {
                    "error": f"Unsupported model variant: {model_variant}"
                }
            }

    def save_artifacts(self, model_results, job_id):
        """Save model artifacts and update database"""
        logger.info(f"Saving artifacts for job ID: {job_id}")
        logger.info(f"Model results: {model_results}")

        # Check if model was trained successfully
        if not model_results.get("model_path"):
            logger.error(f"No model path in results for job {job_id}")
            return model_results

        # In a real implementation, we would copy the model to a more permanent location
        # Here we'll ensure it's saved in the models directory
        import os
        import shutil

        # Ensure the models directory exists
        models_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Get the source model path
        source_path = model_results["model_path"]

        # Define the destination path with job ID for uniqueness
        if os.path.exists(source_path):
            filename = os.path.basename(source_path)
            destination_path = os.path.join(models_dir,
                                            f"job_{job_id}_{filename}")

            # Copy the model file
            try:
                shutil.copy2(source_path, destination_path)
                logger.info(f"Model artifact copied to: {destination_path}")

                # Update the model path in results
                model_results["model_path"] = destination_path
            except Exception as e:
                logger.error(f"Failed to copy model artifact: {str(e)}")
        else:
            logger.error(f"Model file not found at: {source_path}")

        return model_results

    def execute_pipeline(self, config):
        """Execute the entire training pipeline"""
        # Extract configuration
        job_id = config.get('job_id')
        model_type = config.get('model_type')
        model_variant = config.get('model_variant')
        hyperparameters = config.get('hyperparameters')
        mlflow_run_id = config.get('mlflow_run_id')
        mlflow_tracking_uri = config.get('mlflow_tracking_uri')

        try:
            logger.info(
                f"Starting direct training pipeline for job ID: {job_id}")
            logger.info(f"Model type: {model_type}, variant: {model_variant}")

            # Step 1: Load dataset
            dataset_info = self.load_dataset(job_id)

            # Step 2: Train model
            model_results = self.train_model(dataset_info, model_variant,
                                             hyperparameters, mlflow_run_id,
                                             mlflow_tracking_uri)

            # Step 3: Save artifacts
            final_results = self.save_artifacts(model_results, job_id)

            # Step 4: Complete the job
            from ml_pipelines import complete_training_job
            artifacts = [{
                'path': model_results['model_path'],
                'type': 'weights',
                'metrics': model_results['results']
            }]
            complete_training_job(job_id, success=True, artifacts=artifacts)

            logger.info(
                f"Pipeline execution completed successfully for job ID: {job_id}"
            )
            return True, final_results

        except Exception as e:
            logger.exception(f"Error executing pipeline: {str(e)}")
            # Mark job as failed
            from ml_pipelines import complete_training_job
            complete_training_job(job_id, success=False, error_message=str(e))
            return False, {"error": str(e)}


# Create singleton instance
direct_pipeline = DirectTrainingPipeline()


def submit_direct_pipeline(config):
    """
    Submit a training job for direct execution

    Args:
        config: Dictionary with pipeline configuration

    Returns:
        run_id: A unique identifier for this run
    """
    # Generate a unique run ID
    run_id = f"direct-{uuid.uuid4().hex}"

    # Create a copy of the config and add the run ID
    run_config = config.copy()
    run_config['direct_run_id'] = run_id

    # Start the pipeline in a separate thread
    def run_pipeline():
        direct_pipeline.execute_pipeline(run_config)

    thread = threading.Thread(target=run_pipeline)
    thread.daemon = True
    thread.start()

    logger.info(f"Submitted direct training pipeline with run ID: {run_id}")
    return run_id


def train_rf_detr(dataset_path, output_path, hyperparameters, dataset_format='yolo'):
    """
    Train the RF-DETR object detection model

    Args:
        dataset_path: Path to the dataset
        output_path: Path to save the output model
        hyperparameters: Training hyperparameters
        dataset_format: Format of the dataset (yolo or coco)

    Returns:
        Dictionary with training results
    """
    logger.info(f"Training RF-DETR with dataset {dataset_path}, format {dataset_format}")
    logger.info(f"Hyperparameters: {hyperparameters}")

    # Start RF-DETR training
    batch_size = hyperparameters.get('batch_size', 16)
    num_epochs = int(hyperparameters.get('epochs', 10))
    learning_rate = float(hyperparameters.get('learning_rate', 2e-4))
    model_type = hyperparameters.get('model_variant', 'base').lower()

    # Check if dataset format is supported (YOLO or COCO)
    if dataset_format.lower() not in ['yolo', 'coco']:
        logger.warning(f"RF-DETR currently only supports YOLO and COCO formats, but got {dataset_format}")
        return {
            'success': False,
            'error': f"RF-DETR currently only supports YOLO and COCO formats, but got {dataset_format}"
        }

    # Construct dataset paths based on format
    if dataset_format.lower() == 'yolo':
        train_data_path = os.path.join(dataset_path, 'train')
        val_data_path = os.path.join(dataset_path, 'valid')
        if not os.path.exists(val_data_path):
            val_data_path = os.path.join(dataset_path, 'val')
    elif dataset_format.lower() == 'coco':
        train_data_path = os.path.join(dataset_path, 'train')
        val_data_path = os.path.join(dataset_path, 'val')
        # Verifica i file di annotazione COCO
        train_annotations = os.path.join(train_data_path, '_annotations.coco.json')
        val_annotations = os.path.join(val_data_path, '_annotations.coco.json')

        if not os.path.exists(train_annotations):
            logger.error(f"COCO annotations file not found: {train_annotations}")
            return {
                'success': False,
                'error': f"COCO annotations file not found: {train_annotations}"
            }

    # Check if dataset exists
    if not os.path.exists(train_data_path):
        logger.error(f"Training data path not found: {train_data_path}")
        return {
            'success': False,
            'error': f"Training data path not found: {train_data_path}"
        }
