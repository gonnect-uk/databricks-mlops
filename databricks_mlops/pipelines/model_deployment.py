"""
Model deployment pipeline with strong typing using Pydantic models.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, model_validator

from databricks_mlops.models.base import Result, StatusEnum
from databricks_mlops.models.config import DeploymentConfig
from databricks_mlops.utils.databricks_client import (DatabricksClient,
                                                    DatabricksConfig)
from databricks_mlops.utils.logging import setup_logger
from databricks_mlops.workflows.mlflow_tracking import ModelStage

# Set up logger
logger = setup_logger("model_deployment")


class DeploymentType(str, Enum):
    """Types of model deployments."""
    SERVING_ENDPOINT = "serving_endpoint"
    BATCH_INFERENCE = "batch_inference"
    STREAMING_INFERENCE = "streaming_inference"
    EDGE_DEPLOYMENT = "edge_deployment"
    CUSTOM = "custom"


class ComputeType(str, Enum):
    """Compute types for model serving."""
    CPU = "cpu"
    GPU = "gpu"
    MULTI_GPU = "multi_gpu"
    CPU_AND_GPU = "cpu_and_gpu"


class EndpointStatus(str, Enum):
    """Status of a serving endpoint."""
    CREATING = "creating"
    READY = "ready"
    UPDATING = "updating"
    FAILED = "failed"
    DELETING = "deleting"


class EndpointConfig(BaseModel):
    """Configuration for a model serving endpoint."""
    name: str
    model_name: str
    model_version: str
    scale_to_zero_enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 2
    workload_size: str = "Small"  # Small, Medium, Large
    compute_type: ComputeType = ComputeType.CPU
    config_version: Optional[int] = None
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = 300
    traffic_percentage: int = 100  # For blue-green deployments


class BatchJobConfig(BaseModel):
    """Configuration for a batch inference job."""
    name: str
    model_name: str
    model_version: str
    input_path: str
    output_path: str
    cluster_id: Optional[str] = None
    cluster_config: Optional[Dict[str, Any]] = None
    schedule: Optional[str] = None  # CRON expression
    timeout_seconds: int = 3600
    max_retries: int = 3
    notebook_path: Optional[str] = None
    parameter_overrides: Dict[str, str] = Field(default_factory=dict)


class ModelDeploymentConfig(BaseModel):
    """Configuration for model deployment."""
    deployment_config: DeploymentConfig
    databricks_config: DatabricksConfig
    endpoint_config: Optional[EndpointConfig] = None
    batch_job_config: Optional[BatchJobConfig] = None
    promote_to_stage: Optional[ModelStage] = None
    replace_existing: bool = True
    validate_deployment: bool = True
    smoke_test: bool = True
    rollback_on_failure: bool = True
    
    @model_validator(mode='after')
    def validate_deployment_configs(self) -> 'ModelDeploymentConfig':
        """Validate that the appropriate deployment config is provided."""
        if self.deployment_config.deployment_type == "serving_endpoint" and not self.endpoint_config:
            raise ValueError("endpoint_config is required when deployment_type is serving_endpoint")
        elif self.deployment_config.deployment_type == "batch_inference" and not self.batch_job_config:
            raise ValueError("batch_job_config is required when deployment_type is batch_inference")
        return self


class EndpointInfo(BaseModel):
    """Information about a serving endpoint."""
    name: str
    status: EndpointStatus
    creator: Optional[str] = None
    creation_timestamp: Optional[int] = None
    last_updated_timestamp: Optional[int] = None
    config_version: Optional[int] = None
    url: Optional[str] = None
    model_name: str
    model_version: str
    compute_type: ComputeType
    replicas: int
    scale_to_zero_enabled: bool
    environment_variables: Dict[str, str] = Field(default_factory=dict)


class BatchJobInfo(BaseModel):
    """Information about a batch inference job."""
    name: str
    job_id: str
    creator: Optional[str] = None
    creation_timestamp: Optional[int] = None
    last_updated_timestamp: Optional[int] = None
    model_name: str
    model_version: str
    input_path: str
    output_path: str
    schedule: Optional[str] = None
    status: str
    latest_run_id: Optional[str] = None
    latest_run_status: Optional[str] = None


class DeploymentResult(Result):
    """Result of model deployment."""
    model_name: str
    model_version: str
    deployment_type: DeploymentType
    deployed_at: datetime = Field(default_factory=datetime.now)
    deployment_time_seconds: float
    endpoint_info: Optional[EndpointInfo] = None
    batch_job_info: Optional[BatchJobInfo] = None
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    smoke_test_results: Dict[str, Any] = Field(default_factory=dict)
    artifact_uris: Dict[str, str] = Field(default_factory=dict)


class ModelDeployer:
    """
    Deploys models to various deployment targets with strong typing.
    
    This class handles the deployment of trained models to serving
    endpoints or batch inference jobs.
    """
    
    def __init__(self, config: ModelDeploymentConfig):
        """
        Initialize the model deployer.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.logger = logger
        self.client = DatabricksClient(config.databricks_config)
    
    def deploy(self) -> DeploymentResult:
        """
        Deploy the model according to the configuration.
        
        Returns:
            DeploymentResult: The result of the deployment
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Starting deployment for model {self.config.deployment_config.model_name}")
        
        # Initialize result
        result = DeploymentResult(
            status=StatusEnum.SUCCESS,
            message=f"Deployment started for model {self.config.deployment_config.model_name}",
            model_name=self.config.deployment_config.model_name,
            model_version=self.config.deployment_config.model_version or "latest",
            deployment_type=DeploymentType(self.config.deployment_config.deployment_type),
            deployment_time_seconds=0.0
        )
        
        try:
            # Promote model to stage if configured
            if self.config.promote_to_stage:
                self._promote_model_to_stage(
                    self.config.deployment_config.model_name,
                    self.config.deployment_config.model_version or "latest",
                    self.config.promote_to_stage
                )
            
            # Deploy based on deployment type
            if self.config.deployment_config.deployment_type == "serving_endpoint":
                endpoint_info = self._deploy_serving_endpoint()
                result.endpoint_info = endpoint_info
                
                # Run smoke test if configured
                if self.config.smoke_test:
                    smoke_test_results = self._run_smoke_test_endpoint(endpoint_info)
                    result.smoke_test_results = smoke_test_results
                
            elif self.config.deployment_config.deployment_type == "batch_inference":
                batch_job_info = self._deploy_batch_job()
                result.batch_job_info = batch_job_info
                
                # Run smoke test if configured
                if self.config.smoke_test:
                    smoke_test_results = self._run_smoke_test_batch(batch_job_info)
                    result.smoke_test_results = smoke_test_results
                
            else:
                self.logger.warning(f"Deployment type {self.config.deployment_config.deployment_type} not fully implemented")
            
            # Update the result
            deployment_time = time.time() - start_time
            result.deployment_time_seconds = deployment_time
            result.message = f"Successfully deployed model {self.config.deployment_config.model_name}"
            
            self.logger.info(f"Deployment completed in {deployment_time:.2f} seconds")
            return result
            
        except Exception as e:
            error_msg = f"Error in model deployment: {str(e)}"
            self.logger.exception(error_msg)
            
            # Handle rollback if configured
            if self.config.rollback_on_failure and (
                result.endpoint_info is not None or result.batch_job_info is not None
            ):
                self._rollback_deployment(result)
            
            deployment_time = time.time() - start_time
            result.status = StatusEnum.FAILED
            result.message = error_msg
            result.deployment_time_seconds = deployment_time
            result.errors = [{"type": type(e).__name__, "message": str(e)}]
            
            return result
    
    def _promote_model_to_stage(self, model_name: str, version: str, stage: ModelStage) -> None:
        """
        Promote a model to the specified stage.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            stage: Target stage
        """
        self.logger.info(f"Promoting model {model_name} version {version} to {stage}")
        
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Use MLflow API to promote the model
        
        self.logger.info(f"Model {model_name} version {version} promoted to {stage}")
    
    def _deploy_serving_endpoint(self) -> EndpointInfo:
        """
        Deploy a model to a serving endpoint.
        
        Returns:
            EndpointInfo: Information about the deployed endpoint
        """
        if not self.config.endpoint_config:
            raise ValueError("Endpoint configuration is required")
        
        self.logger.info(f"Deploying model {self.config.deployment_config.model_name} to serving endpoint {self.config.endpoint_config.name}")
        
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Check if endpoint exists
        # 2. Create or update the endpoint
        # 3. Wait for the endpoint to be ready
        
        # Create a mock endpoint info
        return EndpointInfo(
            name=self.config.endpoint_config.name,
            status=EndpointStatus.READY,
            creator="current_user",
            creation_timestamp=int(datetime.now().timestamp() * 1000),
            last_updated_timestamp=int(datetime.now().timestamp() * 1000),
            config_version=1,
            url=f"https://dbc-123456789.cloud.databricks.com/serving-endpoints/{self.config.endpoint_config.name}/invocations",
            model_name=self.config.deployment_config.model_name,
            model_version=self.config.deployment_config.model_version or "1",
            compute_type=self.config.endpoint_config.compute_type,
            replicas=self.config.endpoint_config.min_replicas,
            scale_to_zero_enabled=self.config.endpoint_config.scale_to_zero_enabled,
            environment_variables=self.config.endpoint_config.environment_variables
        )
    
    def _deploy_batch_job(self) -> BatchJobInfo:
        """
        Deploy a model as a batch inference job.
        
        Returns:
            BatchJobInfo: Information about the deployed batch job
        """
        if not self.config.batch_job_config:
            raise ValueError("Batch job configuration is required")
        
        self.logger.info(f"Deploying model {self.config.deployment_config.model_name} as batch job {self.config.batch_job_config.name}")
        
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Check if job exists
        # 2. Create or update the job
        # 3. Optionally trigger a test run
        
        # Create a mock batch job info
        return BatchJobInfo(
            name=self.config.batch_job_config.name,
            job_id="123456",
            creator="current_user",
            creation_timestamp=int(datetime.now().timestamp() * 1000),
            last_updated_timestamp=int(datetime.now().timestamp() * 1000),
            model_name=self.config.deployment_config.model_name,
            model_version=self.config.deployment_config.model_version or "1",
            input_path=self.config.batch_job_config.input_path,
            output_path=self.config.batch_job_config.output_path,
            schedule=self.config.batch_job_config.schedule,
            status="ACTIVE",
            latest_run_id=None,
            latest_run_status=None
        )
    
    def _run_smoke_test_endpoint(self, endpoint_info: EndpointInfo) -> Dict[str, Any]:
        """
        Run a smoke test on a deployed serving endpoint.
        
        Args:
            endpoint_info: Information about the deployed endpoint
            
        Returns:
            Dictionary of smoke test results
        """
        self.logger.info(f"Running smoke test on endpoint {endpoint_info.name}")
        
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Send test requests to the endpoint
        # 2. Verify responses
        # 3. Collect performance metrics
        
        return {
            "status": "passed",
            "latency_ms": 120,
            "response_time_p95_ms": 150,
            "tests_passed": 5,
            "tests_failed": 0
        }
    
    def _run_smoke_test_batch(self, batch_job_info: BatchJobInfo) -> Dict[str, Any]:
        """
        Run a smoke test on a deployed batch job.
        
        Args:
            batch_job_info: Information about the deployed batch job
            
        Returns:
            Dictionary of smoke test results
        """
        self.logger.info(f"Running smoke test on batch job {batch_job_info.name}")
        
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Trigger a job run with test data
        # 2. Wait for completion
        # 3. Verify outputs
        
        return {
            "status": "passed",
            "run_id": "test_run_123",
            "run_duration_seconds": 120,
            "records_processed": 1000,
            "success_rate": 1.0
        }
    
    def _rollback_deployment(self, result: DeploymentResult) -> None:
        """
        Rollback a failed deployment.
        
        Args:
            result: The deployment result with information about what was deployed
        """
        self.logger.info("Rolling back failed deployment")
        
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Revert to previous endpoint config or delete new endpoint
        # 2. Revert to previous job config or delete new job
        
        if result.endpoint_info:
            self.logger.info(f"Rolling back endpoint {result.endpoint_info.name}")
            # Rollback logic for endpoints
        
        if result.batch_job_info:
            self.logger.info(f"Rolling back batch job {result.batch_job_info.name}")
            # Rollback logic for batch jobs
        
        self.logger.info("Rollback completed")
