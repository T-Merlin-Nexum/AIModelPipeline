import os
import sys
import logging
import mlflow
import tempfile
import json
from datetime import datetime
from app import app, db
from models import TrainingJob, ModelArtifact

def submit_direct_pipeline(config):
    """
    Submit a training pipeline for direct execution
    
    Args:
        config: Dictionary with pipeline configuration
        
    Returns:
        Run ID for tracking
    """
    # Import the direct training module function
    from direct_training import submit_direct_pipeline as _submit_direct_pipeline
    
    try:
        # Submit to direct training pipeline
        run_id = _submit_direct_pipeline(config)
        logger.info(f"Submitted direct training pipeline with run ID: {run_id}")
        return run_id
    except Exception as e:
        # In case of error, create a simulated ID
        import uuid
        run_id = f"direct-error-{uuid.uuid4().hex}"
        logger.warning(f"Error submitting to direct pipeline: {str(e)}. Using error ID: {run_id}")
        logger.info(f"Pipeline configuration: {config}")
        return run_id

# Initialize logger
logger = logging.getLogger(__name__)

def start_training_job(job_id):
    """Start a training job by creating MLFlow run and submitting to Dagster"""
    with app.app_context():
        job = db.session.get(TrainingJob, job_id)
        if not job:
            logger.error(f"Training job with ID {job_id} not found")
            return False

        # Set up MLFlow tracking
        try:
            mlflow.set_tracking_uri(app.config.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
            experiment_name = f"{job.model_type}-training"
            
            # Create MLFlow experiment if it doesn't exist
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
                
            logger.info(f"MLFLOW_TRACKING_URI experiment {experiment_name} configured with ID {experiment_id}")
            logger.info(f"MLFlow experiment {experiment_name} configured with ID {experiment_id}")
        except Exception as e:
            logger.warning(f"MLFlow setup warning: {str(e)}")
            # Continue without MLFlow tracking if there's an issue
            experiment_id = None
        
        # Start MLFlow run if available
        try:
            if experiment_id:
                mlflow_run = mlflow.start_run(experiment_id=experiment_id, run_name=job.job_name)
                job.mlflow_run_id = mlflow_run.info.run_id
                
                # Log parameters
                hyperparams = job.get_hyperparameters()
                for param_name, param_value in hyperparams.items():
                    mlflow.log_param(param_name, param_value)
                
                mlflow.log_param("model_type", job.model_type)
                mlflow.log_param("model_variant", job.model_variant)
            else:
                # Create a unique run ID if MLFlow is not available
                import uuid
                job.mlflow_run_id = f"direct-{uuid.uuid4().hex}"
        except Exception as e:
            logger.warning(f"MLFlow run creation warning: {str(e)}")
            import uuid
            job.mlflow_run_id = f"direct-{uuid.uuid4().hex}"
        
        # Create pipeline configuration
        config = {
            "job_id": job.id,
            "model_type": job.model_type,
            "model_variant": job.model_variant,
            "hyperparameters": job.get_hyperparameters(),
            "mlflow_run_id": job.mlflow_run_id,
            "mlflow_tracking_uri": app.config.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
        }
        
        # Submit pipeline for direct execution
        run_id = submit_direct_pipeline(config)
        job.dagster_run_id = run_id  # We reuse the same field for compatibility
        
        # Update job status
        job.status = 'running'
        job.started_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Training job {job_id} started with run ID {run_id}")
        return run_id
                    
        # Se non siamo in modalità simulazione, procedi normalmente
        # Set up MLFlow tracking
        mlflow.set_tracking_uri(app.config["MLFLOW_TRACKING_URI"])
        experiment_name = f"{job.model_type}-training"
        
        # Create MLFlow experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.error(f"Error setting up MLFlow experiment: {str(e)}")
            job.status = 'failed'
            job.error_message = f"MLFlow setup error: {str(e)}"
            db.session.commit()
            return False
        
        # Start MLFlow run
        try:
            with mlflow.start_run(experiment_id=experiment_id, run_name=job.job_name) as run:
                # Log parameters
                hyperparams = job.get_hyperparameters()
                for param_name, param_value in hyperparams.items():
                    mlflow.log_param(param_name, param_value)
                
                mlflow.log_param("model_type", job.model_type)
                mlflow.log_param("model_variant", job.model_variant)
                
                # Store MLFlow run ID
                job.mlflow_run_id = run.info.run_id
                
                # Create pipeline configuration file
                config = {
                    "job_id": job.id,
                    "model_type": job.model_type,
                    "model_variant": job.model_variant,
                    "hyperparameters": hyperparams,
                    "mlflow_run_id": run.info.run_id,
                    "mlflow_tracking_uri": app.config["MLFLOW_TRACKING_URI"]
                }
                
                # Submit pipeline for direct execution
                run_id = submit_direct_pipeline(config)
                job.dagster_run_id = run_id  # We reuse the same field for compatibility
                
                # Update job status
                job.status = 'running'
                job.started_at = datetime.utcnow()
                job.error_message = "Esecuzione diretta (MLFlow monitoraggio non disponibile)"
                db.session.commit()
                
                logger.info(f"Training job {job_id} started with MLFlow run ID {run.info.run_id} and Dagster run ID {dagster_run_id}")
                return True
                
        except Exception as e:
            logger.exception(f"Error starting training job: {str(e)}")
            job.status = 'failed'
            job.error_message = str(e)
            db.session.commit()
            return False


def get_job_status(job):
    """Get the latest status of a training job from MLFlow and Dagster"""
    status_data = {
        'status': job.status,
        'started_at': job.started_at.isoformat() if job.started_at else None,
        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
        'error_message': job.error_message,
        'metrics': {},
        'progress': 0.0,
        'model_path': None
    }
    
    # If job is completed or failed, return saved status
    if job.status in ['completed', 'failed', 'cancelled']:
        # Get metrics and model path from artifacts if available
        artifacts = ModelArtifact.query.filter_by(training_job_id=job.id).all()
        for artifact in artifacts:
            if artifact.artifact_type == 'weights':
                status_data['model_path'] = artifact.artifact_path
                
            if artifact.metrics:
                metrics = artifact.get_metrics()
                if metrics:
                    status_data['metrics'] = metrics
                    
                    # If we have metrics, we can calculate total progress
                    status_data['progress'] = 100.0  # Job is complete
        return status_data
    
    # For running jobs, check MLFlow for latest metrics if possible
    if job.mlflow_run_id:
        try:
            # Only try to connect to MLFlow if it's not a direct run ID
            if not job.mlflow_run_id.startswith('direct-'):
                mlflow.set_tracking_uri(app.config.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
                run = mlflow.get_run(job.mlflow_run_id)
                
                # Get metrics from MLFlow
                if run and run.data.metrics:
                    status_data['metrics'] = run.data.metrics
                    logger.info(f"Retrieved metrics from MLFlow: {run.data.metrics}")
                    
                    # Calculate progress if 'epoch' metric is available
                    hyperparams = job.get_hyperparameters()
                    if 'epoch' in run.data.metrics and 'epochs' in hyperparams:
                        current_epoch = run.data.metrics['epoch']
                        total_epochs = hyperparams['epochs']
                        status_data['progress'] = (current_epoch / total_epochs) * 100
                    
                    # Check if expected metrics are present
                    expected_metrics = ['precision', 'recall', 'mAP50', 'mAP50-95']
                    missing_metrics = [m for m in expected_metrics if m not in run.data.metrics]
                    if missing_metrics:
                        logger.warning(f"Some expected metrics are missing in MLFlow: {missing_metrics}")
                        
                    # Get run artifacts
                    try:
                        artifacts = mlflow.list_artifacts(run_id=job.mlflow_run_id)
                        logger.info(f"MLFlow artifacts for run {job.mlflow_run_id}: {artifacts}")
                        
                        # If model artifact is present, update status_data
                        model_artifacts = [a for a in artifacts if a.path.endswith('.pt')]
                        if model_artifacts:
                            logger.info(f"Found model artifacts in MLFlow: {model_artifacts}")
                    except Exception as e:
                        logger.warning(f"Failed to list MLFlow artifacts: {str(e)}")
            else:
                # For direct runs without MLFlow, get progress from job data
                if job.status == 'running':
                    # Calculate progress based on metrics in job artifacts if available
                    artifacts = ModelArtifact.query.filter_by(training_job_id=job.id).all()
                    for artifact in artifacts:
                        if artifact.metrics:
                            metrics = artifact.get_metrics()
                            hyperparams = job.get_hyperparameters()
                            total_epochs = hyperparams.get('epochs', 100)
                            
                            if 'epoch' in metrics and total_epochs:
                                current_epoch = metrics['epoch']
                                progress = (current_epoch / total_epochs) * 100
                                status_data['progress'] = min(progress, 99.0)  # Cap at 99% until completed
                                status_data['metrics'] = metrics
                                break
                    
                    # If no progress info found, use a reasonable estimate
                    if 'metrics' not in status_data or not status_data['metrics']:
                        # Calculate how long the job has been running
                        if job.started_at:
                            elapsed = (datetime.utcnow() - job.started_at).total_seconds()
                            hyperparams = job.get_hyperparameters()
                            total_epochs = hyperparams.get('epochs', 100)
                            
                            # Assume reasonable training time per epoch based on model_type
                            epoch_seconds = 60  # Assume each epoch takes ~1 minute by default
                            if job.model_type == 'yolo':
                                if 'small' in job.model_variant or 's' in job.model_variant:
                                    epoch_seconds = 30
                                elif 'medium' in job.model_variant or 'm' in job.model_variant:
                                    epoch_seconds = 60
                                elif 'large' in job.model_variant or 'l' in job.model_variant:
                                    epoch_seconds = 120
                                elif 'extra' in job.model_variant or 'x' in job.model_variant:
                                    epoch_seconds = 180
                            
                            estimated_epoch = min(int(elapsed / epoch_seconds), total_epochs)
                            progress = (estimated_epoch / total_epochs) * 100
                            
                            status_data['progress'] = min(progress, 99.0)  # Cap at 99% until completed
                            status_data['metrics'] = {
                                'epoch': estimated_epoch,
                                'estimated': True
                            }
        except Exception as e:
            logger.warning(f"Error fetching training metrics: {str(e)}")
            # Continue with default values if there's an error
    
    return status_data


def cancel_training_job(job):
    """Cancel a running training job"""
    try:
        # In a real implementation, we would call Dagster API to cancel the pipeline
        # For demo purposes, we'll just update the job status
        
        # Update job status
        job.status = 'cancelled'
        job.error_message = "Job cancelled by user"
        db.session.commit()
        
        # Cancel MLFlow run (if exists)
        if job.mlflow_run_id:
            mlflow.set_tracking_uri(app.config["MLFLOW_TRACKING_URI"])
            mlflow.end_run(status="KILLED")
        
        logger.info(f"Training job {job.id} cancelled")
        return True
    except Exception as e:
        logger.exception(f"Error cancelling training job: {str(e)}")
        return False


def complete_training_job(job_id, success=True, error_message=None, artifacts=None):
    """Mark a training job as completed or failed"""
    with app.app_context():
        job = db.session.get(TrainingJob, job_id)
        if not job:
            logger.error(f"Training job with ID {job_id} not found")
            return False
        
        job.completed_at = datetime.utcnow()
        
        if success:
            job.status = 'completed'
        else:
            job.status = 'failed'
            job.error_message = error_message
        
        # Try to end MLFlow run if it exists
        if job.mlflow_run_id and not job.mlflow_run_id.startswith('direct-'):
            try:
                mlflow.set_tracking_uri(app.config.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
                
                # Prima verifica se la connessione a MLFlow è possibile
                try:
                    # Ottieni lo stato attuale della run
                    run = mlflow.get_run(job.mlflow_run_id)
                    if run and run.info.lifecycle_stage != "deleted":
                        mlflow.end_run(status="FINISHED" if success else "FAILED")
                        logger.info(f"MLFlow run {job.mlflow_run_id} marked as {'FINISHED' if success else 'FAILED'}")
                    else:
                        logger.warning(f"MLFlow run {job.mlflow_run_id} either doesn't exist or is deleted")
                except Exception as e:
                    # In caso di errore nel get_run, prova comunque a completare la run
                    logger.warning(f"Couldn't get MLFlow run info: {str(e)}, trying to end it anyway")
                    try:
                        # Riattiva la run prima di terminarla
                        mlflow.start_run(run_id=job.mlflow_run_id)
                        mlflow.end_run(status="FINISHED" if success else "FAILED")
                        logger.info(f"MLFlow run {job.mlflow_run_id} marked as FINISHED")
                    except Exception as ex:
                        logger.warning(f"Failed final attempt to end MLFlow run: {str(ex)}")
                        import traceback
                        logger.warning(f"MLFlow end_run error details: {traceback.format_exc()}")
            except Exception as e:
                logger.warning(f"Failed to end MLFlow run: {str(e)}")
        
        # Save artifacts if provided
        if artifacts and success:
            for artifact_info in artifacts:
                if 'path' not in artifact_info or not artifact_info['path']:
                    logger.warning(f"Skipping artifact with missing path: {artifact_info}")
                    continue
                    
                artifact = ModelArtifact(
                    training_job_id=job_id,
                    artifact_path=artifact_info['path'],
                    artifact_type=artifact_info.get('type', 'unknown')
                )
                
                if 'metrics' in artifact_info:
                    artifact.set_metrics(artifact_info['metrics'])
                
                db.session.add(artifact)
                logger.info(f"Added artifact: {artifact_info['path']} of type {artifact_info.get('type', 'unknown')}")
        
        db.session.commit()
        logger.info(f"Training job {job_id} marked as {'completed' if success else 'failed'}")
        return True
