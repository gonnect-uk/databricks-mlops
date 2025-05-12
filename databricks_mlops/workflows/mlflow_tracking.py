"""
Strongly-typed MLflow integration for model tracking and registry operations.
"""
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
from mlflow.entities import Run
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field, field_validator, model_validator

from databricks_mlops.models.base import ArtifactMetadata, Result, StatusEnum
from databricks_mlops.utils.logging import setup_logger

# Create a dedicated logger
logger = setup_logger("mlflow_tracking")


class TrackingError(Exception):
    """Specialized exception for MLflow tracking errors."""
    pass


class ModelRegistryError(Exception):
    """Specialized exception for MLflow model registry errors."""
    pass


class ModelStage(str, Enum):
    """Strongly-typed model stages in MLflow."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelVersionStatus(str, Enum):
    """Strongly-typed model version status in MLflow."""
    PENDING_REGISTRATION = "PENDING_REGISTRATION"
    FAILED_REGISTRATION = "FAILED_REGISTRATION"
    READY = "READY"


class RunStatus(str, Enum):
    """Strongly-typed run status in MLflow."""
    RUNNING = "RUNNING"
    SCHEDULED = "SCHEDULED"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


class MetricValue(BaseModel):
    """Strongly-typed metric value in MLflow."""
    key: str
    value: float
    timestamp: int
    step: int = 0


class ParamValue(BaseModel):
    """Strongly-typed parameter value in MLflow."""
    key: str
    value: str


class TagValue(BaseModel):
    """Strongly-typed tag value in MLflow."""
    key: str
    value: str


class RunInfo(BaseModel):
    """Strongly-typed run information."""
    run_id: str
    experiment_id: str
    run_name: Optional[str] = None
    status: RunStatus
    start_time: int
    end_time: Optional[int] = None
    artifact_uri: str
    lifecycle_stage: str
    user_id: Optional[str] = None


class RunData(BaseModel):
    """Strongly-typed run data in MLflow."""
    metrics: Dict[str, float] = Field(default_factory=dict)
    params: Dict[str, str] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)


class RunDetail(BaseModel):
    """Strongly-typed run detail in MLflow."""
    info: RunInfo
    data: RunData


class ModelVersion(BaseModel):
    """Strongly-typed model version in MLflow."""
    name: str
    version: str
    creation_timestamp: int
    last_updated_timestamp: int
    description: Optional[str] = None
    user_id: Optional[str] = None
    current_stage: ModelStage = ModelStage.NONE
    source: str
    run_id: Optional[str] = None
    status: ModelVersionStatus = ModelVersionStatus.READY
    status_message: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)


class TrackingConfig(BaseModel):
    """Configuration for MLflow tracking."""
    tracking_uri: Optional[str] = None
    registry_uri: Optional[str] = None
    experiment_name: str
    run_name: Optional[str] = None
    artifact_location: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    auto_log_params: bool = True
    auto_log_metrics: bool = True
    auto_log_artifacts: bool = True
    nested: bool = False
    tags: Dict[str, str] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_model_version(self) -> 'TrackingConfig':
        """Validate that model_version is only set if model_name is set."""
        if self.model_version is not None and self.model_name is None:
            raise ValueError("model_version can only be set if model_name is set")
        return self


class RegisterModelResult(Result):
    """Strongly-typed result of registering a model in MLflow."""
    model_name: str
    model_version: str
    model_source: str
    run_id: Optional[str] = None
    registry_url: Optional[str] = None


class MLflowTracker:
    """
    Strongly-typed wrapper for MLflow tracking and registry operations.
    
    This class provides a Pydantic-driven interface to MLflow functionality
    with proper error handling and type safety.
    """
    
    def __init__(self, config: TrackingConfig):
        """
        Initialize the MLflow tracker.
        
        Args:
            config: MLflow tracking configuration
        """
        self.config = config
        self.logger = logger
        self.active_run: Optional[mlflow.ActiveRun] = None
        self.experiment_id: Optional[str] = None
        self.client = self._setup_client()
    
    def _setup_client(self) -> MlflowClient:
        """
        Set up the MLflow client with the configured tracking and registry URIs.
        
        Returns:
            MlflowClient: Configured MLflow client
        """
        try:
            # Set tracking URI if provided
            if self.config.tracking_uri:
                mlflow.set_tracking_uri(self.config.tracking_uri)
                self.logger.info(f"Set MLflow tracking URI to {self.config.tracking_uri}")
            
            # Set registry URI if provided
            if self.config.registry_uri:
                mlflow.set_registry_uri(self.config.registry_uri)
                self.logger.info(f"Set MLflow registry URI to {self.config.registry_uri}")
            
            # Create the client
            return MlflowClient()
            
        except Exception as e:
            error_msg = f"Failed to setup MLflow client: {str(e)}"
            self.logger.error(error_msg)
            raise TrackingError(error_msg) from e
    
    def start_run(self, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start an MLflow run with the specified tags.
        
        Args:
            tags: Additional tags to set on the run
            
        Returns:
            str: The ID of the started run
            
        Raises:
            TrackingError: If the run cannot be started
        """
        try:
            # Create or get the experiment
            experiment = self._get_or_create_experiment(
                name=self.config.experiment_name,
                artifact_location=self.config.artifact_location
            )
            self.experiment_id = experiment.experiment_id
            
            # Combine configuration tags with provided tags
            combined_tags = {**self.config.tags, **(tags or {})}
            
            # Start the run
            self.active_run = mlflow.start_run(
                run_name=self.config.run_name,
                experiment_id=self.experiment_id,
                nested=self.config.nested,
                tags=combined_tags
            )
            
            run_id = self.active_run.info.run_id
            self.logger.info(f"Started MLflow run with ID: {run_id}")
            
            return run_id
            
        except Exception as e:
            error_msg = f"Failed to start MLflow run: {str(e)}"
            self.logger.error(error_msg)
            raise TrackingError(error_msg) from e
    
    def _get_or_create_experiment(self, name: str, artifact_location: Optional[str] = None) -> mlflow.entities.Experiment:
        """
        Get an existing experiment or create a new one if it doesn't exist.
        
        Args:
            name: The name of the experiment
            artifact_location: Optional artifact location for the experiment
            
        Returns:
            Experiment: The retrieved or created experiment
            
        Raises:
            TrackingError: If the experiment cannot be created or retrieved
        """
        try:
            # Try to get the experiment by name
            experiment = mlflow.get_experiment_by_name(name)
            
            # If the experiment doesn't exist, create it
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=name,
                    artifact_location=artifact_location
                )
                experiment = mlflow.get_experiment(experiment_id)
                self.logger.info(f"Created new experiment '{name}' with ID: {experiment_id}")
            else:
                self.logger.info(f"Using existing experiment '{name}' with ID: {experiment.experiment_id}")
            
            return experiment
            
        except Exception as e:
            error_msg = f"Failed to get or create experiment '{name}': {str(e)}"
            self.logger.error(error_msg)
            raise TrackingError(error_msg) from e
    
    def log_param(self, key: str, value: Any) -> None:
        """
        Log a parameter to the current MLflow run.
        
        Args:
            key: The parameter name
            value: The parameter value (will be converted to string)
            
        Raises:
            TrackingError: If the parameter cannot be logged
        """
        try:
            # Ensure there's an active run
            if not self.active_run:
                raise TrackingError("No active MLflow run")
            
            # Log the parameter
            mlflow.log_param(key, value)
            self.logger.debug(f"Logged parameter {key}={value}")
            
        except TrackingError:
            raise
        except Exception as e:
            error_msg = f"Failed to log parameter {key}: {str(e)}"
            self.logger.error(error_msg)
            raise TrackingError(error_msg) from e
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log multiple parameters to the current MLflow run.
        
        Args:
            params: Dictionary of parameter names and values
            
        Raises:
            TrackingError: If the parameters cannot be logged
        """
        try:
            # Ensure there's an active run
            if not self.active_run:
                raise TrackingError("No active MLflow run")
            
            # Log all parameters
            mlflow.log_params(params)
            self.logger.debug(f"Logged {len(params)} parameters")
            
        except TrackingError:
            raise
        except Exception as e:
            error_msg = f"Failed to log parameters: {str(e)}"
            self.logger.error(error_msg)
            raise TrackingError(error_msg) from e
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric to the current MLflow run.
        
        Args:
            key: The metric name
            value: The metric value
            step: Optional step index for the metric
            
        Raises:
            TrackingError: If the metric cannot be logged
        """
        try:
            # Ensure there's an active run
            if not self.active_run:
                raise TrackingError("No active MLflow run")
            
            # Log the metric
            mlflow.log_metric(key, value, step=step)
            self.logger.debug(f"Logged metric {key}={value} at step={step}")
            
        except TrackingError:
            raise
        except Exception as e:
            error_msg = f"Failed to log metric {key}: {str(e)}"
            self.logger.error(error_msg)
            raise TrackingError(error_msg) from e
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics to the current MLflow run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step index for the metrics
            
        Raises:
            TrackingError: If the metrics cannot be logged
        """
        try:
            # Ensure there's an active run
            if not self.active_run:
                raise TrackingError("No active MLflow run")
            
            # Log all metrics
            mlflow.log_metrics(metrics, step=step)
            self.logger.debug(f"Logged {len(metrics)} metrics at step={step}")
            
        except TrackingError:
            raise
        except Exception as e:
            error_msg = f"Failed to log metrics: {str(e)}"
            self.logger.error(error_msg)
            raise TrackingError(error_msg) from e
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> ArtifactMetadata:
        """
        Log an artifact to the current MLflow run.
        
        Args:
            local_path: Path to the local file to log
            artifact_path: Optional path within the artifact directory
            
        Returns:
            ArtifactMetadata: Metadata for the logged artifact
            
        Raises:
            TrackingError: If the artifact cannot be logged
        """
        try:
            # Ensure there's an active run
            if not self.active_run:
                raise TrackingError("No active MLflow run")
            
            # Log the artifact
            mlflow.log_artifact(local_path, artifact_path)
            
            # Create metadata
            artifact_name = local_path.split('/')[-1]
            artifact_path_str = artifact_path or ""
            full_path = f"{artifact_path_str}/{artifact_name}" if artifact_path else artifact_name
            
            metadata = ArtifactMetadata(
                name=artifact_name,
                artifact_type="file",
                path=full_path,
                created_at=datetime.now()
            )
            
            self.logger.debug(f"Logged artifact {local_path} to {full_path}")
            return metadata
            
        except TrackingError:
            raise
        except Exception as e:
            error_msg = f"Failed to log artifact {local_path}: {str(e)}"
            self.logger.error(error_msg)
            raise TrackingError(error_msg) from e
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> List[ArtifactMetadata]:
        """
        Log all artifacts in a directory to the current MLflow run.
        
        Args:
            local_dir: Path to the local directory containing artifacts
            artifact_path: Optional path within the artifact directory
            
        Returns:
            List[ArtifactMetadata]: Metadata for the logged artifacts
            
        Raises:
            TrackingError: If the artifacts cannot be logged
        """
        try:
            # Ensure there's an active run
            if not self.active_run:
                raise TrackingError("No active MLflow run")
            
            # Log all artifacts in the directory
            mlflow.log_artifacts(local_dir, artifact_path)
            
            # This is a simplified approach as we don't have direct access to the artifact list
            # In a real implementation, you might want to scan the directory to get accurate metadata
            
            metadata = ArtifactMetadata(
                name=local_dir.split('/')[-1],
                artifact_type="directory",
                path=artifact_path or "",
                created_at=datetime.now()
            )
            
            self.logger.debug(f"Logged artifacts from directory {local_dir} to {artifact_path or '/'}")
            return [metadata]
            
        except TrackingError:
            raise
        except Exception as e:
            error_msg = f"Failed to log artifacts from {local_dir}: {str(e)}"
            self.logger.error(error_msg)
            raise TrackingError(error_msg) from e
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        pip_requirements: Optional[List[str]] = None,
        code_paths: Optional[List[str]] = None,
        **kwargs
    ) -> ArtifactMetadata:
        """
        Log a model to the current MLflow run.
        
        Args:
            model: The model to log
            artifact_path: Path within the artifact directory
            pip_requirements: Optional pip requirements for the model
            code_paths: Optional paths to code files to include
            **kwargs: Additional arguments for the mlflow.log_model function
            
        Returns:
            ArtifactMetadata: Metadata for the logged model
            
        Raises:
            TrackingError: If the model cannot be logged
        """
        try:
            # Ensure there's an active run
            if not self.active_run:
                raise TrackingError("No active MLflow run")
            
            # Log the model
            model_info = mlflow.log_model(
                model,
                artifact_path=artifact_path,
                pip_requirements=pip_requirements,
                code_paths=code_paths,
                **kwargs
            )
            
            metadata = ArtifactMetadata(
                name=artifact_path,
                artifact_type="model",
                path=model_info.model_uri,
                created_at=datetime.now(),
                version=kwargs.get("version"),
            )
            
            self.logger.info(f"Logged model to {model_info.model_uri}")
            return metadata
            
        except TrackingError:
            raise
        except Exception as e:
            error_msg = f"Failed to log model: {str(e)}"
            self.logger.error(error_msg)
            raise TrackingError(error_msg) from e
    
    def register_model(
        self,
        model_uri: str,
        name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> RegisterModelResult:
        """
        Register a model in the MLflow Model Registry.
        
        Args:
            model_uri: URI to the model, can be a run URI or a local path
            name: Name to register the model under (if None, uses config.model_name)
            tags: Optional tags to set on the model version
            
        Returns:
            RegisterModelResult: The result of the registration
            
        Raises:
            ModelRegistryError: If the model cannot be registered
        """
        try:
            # Determine the model name to use
            model_name = name or self.config.model_name
            if not model_name:
                raise ModelRegistryError("No model name provided")
            
            # Register the model
            model_details = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )
            
            # Create a successful result
            result = RegisterModelResult(
                status=StatusEnum.SUCCESS,
                message=f"Successfully registered model '{model_name}' as version {model_details.version}",
                model_name=model_name,
                model_version=model_details.version,
                model_source=model_uri,
                run_id=self.active_run.info.run_id if self.active_run else None,
                registry_url=f"{mlflow.get_registry_uri()}/models/{model_name}/versions/{model_details.version}" if mlflow.get_registry_uri() else None
            )
            
            self.logger.info(f"Registered model '{model_name}' as version {model_details.version} from {model_uri}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to register model: {str(e)}"
            self.logger.error(error_msg)
            
            # Create a failure result
            return RegisterModelResult(
                status=StatusEnum.FAILED,
                message=f"Failed to register model: {str(e)}",
                model_name=name or self.config.model_name or "unknown",
                model_version="unknown",
                model_source=model_uri,
                errors=[{"type": type(e).__name__, "message": str(e)}]
            )
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.
        
        Args:
            status: The status to set for the run ("FINISHED", "FAILED", "KILLED")
            
        Raises:
            TrackingError: If the run cannot be ended
        """
        try:
            # Only end the run if there is an active run
            if self.active_run:
                mlflow.end_run(status)
                self.logger.info(f"Ended MLflow run with status: {status}")
                self.active_run = None
                
        except Exception as e:
            error_msg = f"Failed to end MLflow run: {str(e)}"
            self.logger.error(error_msg)
            raise TrackingError(error_msg) from e
    
    def get_run(self, run_id: str) -> RunDetail:
        """
        Get details for a specific MLflow run.
        
        Args:
            run_id: The ID of the run to retrieve
            
        Returns:
            RunDetail: Details of the run
            
        Raises:
            TrackingError: If the run cannot be retrieved
        """
        try:
            # Get the run from MLflow
            run = self.client.get_run(run_id)
            
            # Convert to strongly-typed models
            run_info = RunInfo(
                run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
                run_name=run.data.tags.get("mlflow.runName"),
                status=RunStatus(run.info.status),
                start_time=run.info.start_time,
                end_time=run.info.end_time,
                artifact_uri=run.info.artifact_uri,
                lifecycle_stage=run.info.lifecycle_stage,
                user_id=run.info.user_id
            )
            
            run_data = RunData(
                metrics={k: v for k, v in run.data.metrics.items()},
                params={k: v for k, v in run.data.params.items()},
                tags={k: v for k, v in run.data.tags.items()}
            )
            
            return RunDetail(info=run_info, data=run_data)
            
        except Exception as e:
            error_msg = f"Failed to get run {run_id}: {str(e)}"
            self.logger.error(error_msg)
            raise TrackingError(error_msg) from e
    
    def transition_model_version_stage(
        self,
        model_name: str,
        version: str,
        stage: ModelStage,
        archive_existing_versions: bool = False
    ) -> Result:
        """
        Transition a model version to a different stage.
        
        Args:
            model_name: The name of the registered model
            version: The version of the model to transition
            stage: The target stage
            archive_existing_versions: Whether to archive existing versions in the target stage
            
        Returns:
            Result: The result of the transition operation
            
        Raises:
            ModelRegistryError: If the transition cannot be performed
        """
        try:
            # Transition the model version
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage.value,
                archive_existing_versions=archive_existing_versions
            )
            
            # Create a successful result
            result = Result(
                status=StatusEnum.SUCCESS,
                message=f"Successfully transitioned model '{model_name}' version {version} to stage '{stage.value}'",
                details={
                    "model_name": model_name,
                    "version": version,
                    "stage": stage.value,
                    "archived_existing_versions": archive_existing_versions
                }
            )
            
            self.logger.info(f"Transitioned model '{model_name}' version {version} to stage '{stage.value}'")
            return result
            
        except Exception as e:
            error_msg = f"Failed to transition model version: {str(e)}"
            self.logger.error(error_msg)
            
            # Create a failure result
            return Result(
                status=StatusEnum.FAILED,
                message=f"Failed to transition model version: {str(e)}",
                errors=[{"type": type(e).__name__, "message": str(e)}]
            )
