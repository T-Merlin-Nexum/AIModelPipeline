import logging
import os
import shutil  # Added for copying files
import traceback  # Added for detailed error logging

import mlflow
import yaml  # Added for dataset YAML creation
from ultralytics import YOLO, settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_yolo_model(dataset_info, model_variant, hyperparameters, mlflow_run_id, mlflow_tracking_uri):
    """Train YOLO model with provided configuration"""
    logger.info(f"Training YOLO model variant: {model_variant} with MLFlow run ID: {mlflow_run_id}")
    logger.info(f"Using dataset: {dataset_info}")
    logger.info(f"Hyperparameters: {hyperparameters}")

    # Ensure model_variant has .pt extension for YOLO class to auto-download
    if not model_variant.endswith(".pt"):
        model_weights_identifier = f"{model_variant}.pt"
        logger.info(f"Model variant '{model_variant}' adjusted to '{model_weights_identifier}' for Ultralytics YOLO.")
    else:
        model_weights_identifier = model_variant

    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created models directory: {models_dir}")

    # Determine model path for saving the trained model
    # Use the original model_variant (without .pt) for a cleaner filename if preferred
    base_model_name = model_variant.replace(".pt", "")
    model_filename = f"{base_model_name}_{mlflow_run_id[:8]}_trained.pt"
    trained_model_output_path = os.path.join(models_dir, model_filename)

    # Get training parameters
    total_epochs = int(hyperparameters.get('epochs', 100))
    batch_size = int(hyperparameters.get('batch_size', 16))
    img_size = int(hyperparameters.get('img_size', 640))
    learning_rate = float(hyperparameters.get('learning_rate', 0.01))
    patience = int(hyperparameters.get('patience', 50))
    project_name = hyperparameters.get('project', 'training_jobs')
    experiment_name = hyperparameters.get('name', f"job_{mlflow_run_id[:8]}")
    cache_dataset = hyperparameters.get('cache', False)
    num_workers = int(hyperparameters.get('workers', 1))  # Defaulting to 1 for potentially lower memory usage

    # Get real dataset path
    dataset_path = dataset_info.get('dataset_path')
    logger.info(f"Starting YOLO training on dataset: {dataset_path} for {total_epochs} epochs")

    # Connect to MLFlow for logging if available
    mlflow_active = False
    if mlflow_tracking_uri:
        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            # Ensure the experiment exists or is created
            experiment = mlflow.get_experiment_by_name(project_name)
            if experiment is None:
                mlflow.create_experiment(project_name)
            mlflow.set_experiment(project_name)

            mlflow_active = True
            logger.info(f"MLFlow logging enabled. Tracking URI: {mlflow_tracking_uri}, Experiment: {project_name}")
        except Exception as e:
            logger.warning(f"MLFlow logging disabled: {str(e)}")
    else:
        logger.info("MLFlow tracking URI not provided. MLFlow logging will be skipped.")

    try:
        logger.info(f"Initializing YOLO model with identifier: '{model_weights_identifier}'")
        # YOLO class will download weights if model_weights_identifier is a known pretrained model like 'yolov8n.pt'
        # and it's not found locally in the expected ultralytics cache.
        model = YOLO(model_weights_identifier)
        logger.info(f"Successfully initialized YOLO model: {model_weights_identifier}")

        settings.update({'mlflow': mlflow_active})  # Update ultralytics settings for MLflow

        # Configure dataset
        if not dataset_path or not os.path.exists(dataset_path):
            logger.warning(
                f"Dataset path '{dataset_path}' not found or not provided. Using COCO8 example dataset for training.")
            dataset_path = "coco8.yaml"  # Ultralytics will download this if not present
        elif os.path.isdir(dataset_path):
            yaml_path = os.path.join(dataset_path, "data.yaml")
            if os.path.exists(yaml_path):
                logger.info(f"Using existing dataset YAML configuration: {yaml_path}")
                dataset_path = yaml_path
            else:
                logger.info("Dataset path is a directory. Attempting to create a temporary data.yaml.")
                train_dir = os.path.join(dataset_path, "train")
                valid_dir = os.path.join(dataset_path, "valid")
                test_dir = os.path.join(dataset_path, "test")  # Optional

                # Check for images and labels subdirectories
                train_images_exist = os.path.exists(os.path.join(train_dir, "images")) if os.path.exists(
                    train_dir) else False
                train_labels_exist = os.path.exists(os.path.join(train_dir, "labels")) if os.path.exists(
                    train_dir) else False
                valid_images_exist = os.path.exists(os.path.join(valid_dir, "images")) if os.path.exists(
                    valid_dir) else False
                valid_labels_exist = os.path.exists(os.path.join(valid_dir, "labels")) if os.path.exists(
                    valid_dir) else False

                yaml_config = {
                    "path": os.path.abspath(dataset_path),  # Absolute path is often more robust
                    "train": os.path.join("train", "images") if train_images_exist else "train",
                    "val": os.path.join("valid", "images") if valid_images_exist else "valid",
                }
                if os.path.exists(test_dir) and os.path.exists(os.path.join(test_dir, "images")):
                    yaml_config["test"] = os.path.join("test", "images")
                elif os.path.exists(test_dir):  # if test dir exists but not test/images
                    yaml_config["test"] = "test"

                # Auto-detect classes from labels in the training set
                detected_classes = {}
                if train_labels_exist:
                    labels_dir_path = os.path.join(train_dir, "labels")
                    class_ids = set()
                    label_files = [f for f in os.listdir(labels_dir_path) if f.endswith('.txt')]
                    for label_file in label_files[:max(20, len(label_files))]:  # Scan a sample of files
                        try:
                            with open(os.path.join(labels_dir_path, label_file), 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if parts and parts[0].isdigit():
                                        class_ids.add(int(parts[0]))
                        except Exception as e:
                            logger.warning(f"Could not read or parse label file {label_file}: {e}")

                    if class_ids:
                        for class_id in sorted(list(class_ids)):
                            detected_classes[class_id] = f"class_{class_id}"
                        logger.info(f"Auto-detected classes: {detected_classes}")
                    else:
                        logger.warning(
                            "No classes detected from label files. MLFLow might not function correctly without them.")

                if not detected_classes:  # Fallback if no classes are detected
                    logger.warning(
                        "No classes auto-detected. Using placeholder names: {0: 'object'}. Please verify your dataset structure and labels.")
                    yaml_config["names"] = {0: "object"}
                else:
                    yaml_config["names"] = detected_classes

                # Save the YAML file
                generated_yaml_path = os.path.join(dataset_path, "autogenerated_data.yaml")
                with open(generated_yaml_path, 'w') as f:
                    yaml.dump(yaml_config, f, sort_keys=False, default_flow_style=None)
                logger.info(f"Dataset YAML configuration created at: {generated_yaml_path}")
                dataset_path = generated_yaml_path
        elif not dataset_path.endswith(('.yaml', '.yml')):
            logger.error(
                f"Dataset path '{dataset_path}' is a file but not a YAML configuration. Please provide a directory or a .yaml file.")
            raise ValueError("Invalid dataset path format.")

        logger.info(f"Starting YOLO training with dataset: {dataset_path}")
        logger.info(f"Epochs: {total_epochs}, Batch size: {batch_size}, Image size: {img_size}")
        logger.info(f"Learning rate: {learning_rate}, Patience: {patience}")
        logger.info(f"Project: {project_name}, Experiment Name: {experiment_name}")
        logger.info(f"Cache dataset: {cache_dataset}, Workers: {num_workers}")

        # Start MLflow run if active
        active_mlflow_run = None
        if mlflow_active:
            try:
                active_mlflow_run = mlflow.start_run(run_id=mlflow_run_id, run_name=experiment_name, nested=True)
                logger.info(f"Started MLFlow run with ID: {active_mlflow_run.info.run_id} and name: {experiment_name}")
                # Log hyperparameters
                mlflow.log_params({
                    "model_variant": model_variant,
                    "epochs": total_epochs,
                    "batch_size": batch_size,
                    "img_size": img_size,
                    "learning_rate": learning_rate,
                    "patience": patience,
                    "dataset": dataset_path,
                    "project_ultralytics": project_name,  # ultralytics project
                    "name_ultralytics": experiment_name  # ultralytics experiment name
                })
            except Exception as e:
                logger.warning(
                    f"Failed to start MLFlow run or log parameters: {str(e)}. Training will continue without MLFlow logging for this run.")
                mlflow_active = False  # Disable further MLflow attempts for this training

        logger.warning(f"DATASET PATH: {dataset_path}")

        # Train the model
        results = model.train(
            data=dataset_path,
            epochs=total_epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=learning_rate,
            patience=patience,
            save=True,  # Ultralytics saves checkpoints and final model
            project=project_name,  # Ultralytics' own project folder
            name=experiment_name,  # Ultralytics' own run name folder
            exist_ok=True,  # Allow re-running into the same folder
            cache=cache_dataset,
            workers=num_workers,
            # Other parameters can be added here e.g. device, optimizer etc.
        )

        logger.info(f"Training completed. Results object: {results}")
        logger.info(f"Metrics keys available: {results.results_dict.keys()}")

        # The 'best.pt' model is saved by ultralytics in project/name/weights/best.pt
        # The 'last.pt' model is also available.
        ultralytics_output_dir = os.path.join(os.getcwd(), project_name, experiment_name)
        best_model_from_ultralytics = os.path.join(ultralytics_output_dir, "weights", "best.pt")

        if os.path.exists(best_model_from_ultralytics):
            shutil.copy2(best_model_from_ultralytics, trained_model_output_path)
            logger.info(f"Best trained model copied to: {trained_model_output_path}")
        else:
            logger.warning(f"Could not find 'best.pt' at {best_model_from_ultralytics}. Check training output.")
            # Fallback: try to save the 'last' model if available, or the current model state.
            last_model_from_ultralytics = os.path.join(ultralytics_output_dir, "weights", "last.pt")
            if os.path.exists(last_model_from_ultralytics):
                shutil.copy2(last_model_from_ultralytics, trained_model_output_path)
                logger.info(f"Last trained model copied to: {trained_model_output_path}")
            else:
                model.save(trained_model_output_path)  # Save current state of the model object
                logger.info(f"Saved current model state to: {trained_model_output_path}")

        # Log metrics to MLFlow
        final_metrics = results.box

        # Ultralytics might also log epochs if training completes fully via its own callback
        # We can log the requested epochs or try to find completed epochs
        epochs_completed = results.epoch if hasattr(results,
                                                    'epoch') and results.epoch is not None else total_epochs
        metrics_to_log = {
            "precision": final_metrics.mp,
            "recall": final_metrics.mr,
            "mAP50": final_metrics.map50,
            "mAP50-95": final_metrics.map,
            "epochs_completed": epochs_completed,
            "fitness": final_metrics.fitness()
        }

        logger.info(f"Logging final metrics to MLFlow: {metrics_to_log}")
        for metric_name, metric_value in metrics_to_log.items():
            try:
                mlflow.log_metric(metric_name, float(metric_value))
            except Exception as e:
                logger.warning(f"Failed to log metric {metric_name}={metric_value} to MLFlow: {str(e)}")

        # Log model artifact
        if os.path.exists(trained_model_output_path):
            mlflow.log_artifact(trained_model_output_path, artifact_path="model")
            logger.info(f"Trained model artifact logged to MLFlow: {trained_model_output_path}")
        else:
            logger.warning(f"Trained model file {trained_model_output_path} not found for MLFlow logging.")

        # Log plot images and other artifacts generated by Ultralytics
        if os.path.exists(ultralytics_output_dir):
            # Log common result files/plots
            files_to_log = [
                "results.csv", "confusion_matrix.png", "F1_curve.png",
                "P_curve.png", "PR_curve.png", "R_curve.png",
                "labels.jpg", "labels_correlogram.jpg",
                "val_batch0_labels.jpg", "val_batch0_pred.jpg"  # example validation batch
            ]
            for f_name in files_to_log:
                f_path = os.path.join(ultralytics_output_dir, f_name)
                if os.path.exists(f_path):
                    try:
                        mlflow.log_artifact(f_path, artifact_path="training_plots")
                        logger.info(f"Logged artifact {f_name} to MLFlow.")
                    except Exception as e:
                        logger.warning(f"Failed to log artifact {f_path} to MLFlow: {e}")
            # Log all PNGs and JPGs if not already covered
            for root, _, files in os.walk(ultralytics_output_dir):
                for file in files:
                    if file.endswith(('.png', '.jpg')) and file not in files_to_log:
                        img_path = os.path.join(root, file)
                        rel_path = os.path.relpath(root, ultralytics_output_dir)
                        artifact_sub_path = os.path.join("training_plots",
                                                         rel_path) if rel_path != '.' else "training_plots"
                        try:
                            mlflow.log_artifact(img_path, artifact_path=artifact_sub_path)
                            logger.info(f"Logged plot to MLFlow: {img_path} under {artifact_sub_path}")
                        except Exception as e:
                            logger.warning(f"Failed to log plot {img_path} to MLFlow: {e}")
        logger.info("MLFlow logging for metrics and artifacts completed.")

        # Return training results
        return_results = {
            "model_path": trained_model_output_path if os.path.exists(trained_model_output_path) else None,
            "results": {
                "precision": final_metrics.mp,
                "recall": final_metrics.mr,
                "mAP50": final_metrics.map50,
                "mAP50-95": final_metrics.map,
                "epochs_completed": epochs_completed,
                "fitness": final_metrics.fitness()
            },
            "ultralytics_output_dir": ultralytics_output_dir
        }

    except Exception as e:
        logger.error(f"Error during YOLO training pipeline: {str(e)}")
        logger.error(f"Detailed traceback: {traceback.format_exc()}")
        # Fallback: if an initial model identifier was provided (implying pretrained),
        # try to report that as the model path, though training failed.
        # Ultralytics might have downloaded it to its cache.
        # This part is tricky as we don't have a specific 'pretrained_weights_path' anymore.
        # The best we can do is signal failure.
        return_results = {
            "model_path": None,  # Training failed, so no new trained model path
            "results": {
                "error": str(e),
                "info": f"Training failed for model variant {model_variant}. Check logs for details. Attempted to use '{model_weights_identifier}' as base."
            },
            "ultralytics_output_dir": None
        }
    finally:
        if mlflow_active and active_mlflow_run:
            try:
                mlflow.end_run()
                logger.info(f"Ended MLFlow run: {active_mlflow_run.info.run_id}")
            except Exception as e:
                logger.warning(f"Error ending MLFlow run: {str(e)}")

    return return_results
