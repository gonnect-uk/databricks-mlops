"""
Strongly-typed Databricks SDK client for the MLOps framework.
"""
import logging
from typing import Any, Dict, List, Optional, Union

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from pydantic import BaseModel, Field, field_validator

from databricks_mlops.models.base import Result, StatusEnum


class DatabricksConfig(BaseModel):
    """Configuration for Databricks SDK authentication."""
    host: str
    token: Optional[str] = None
    profile: Optional[str] = None
    cluster_id: Optional[str] = None
    job_id: Optional[str] = None
    use_azure_cli: bool = False
    use_databricks_cli: bool = True
    
    class Config:
        """Pydantic configuration for sensitive values."""
        protected_namespaces = ()


class JobRunParams(BaseModel):
    """Parameters for running Databricks jobs with strong typing."""
    job_id: Optional[str] = None
    job_name: Optional[str] = None
    notebook_path: Optional[str] = None
    python_file_path: Optional[str] = None
    parameters: Dict[str, str] = Field(default_factory=dict)
    cluster_id: Optional[str] = None
    timeout_seconds: int = 3600
    idempotency_token: Optional[str] = None
    
    @field_validator("job_id", "job_name")
    @classmethod
    def validate_job_identifier(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate that either job_id or job_name is provided."""
        # If this is validating job_id and job_name is already set, job_id can be None
        if "job_name" in values and values["job_name"] and v is None and values["_field_name"] == "job_id":
            return None
            
        # If this is validating job_name and job_id is already set, job_name can be None
        if "_field_name" in values and values["_field_name"] == "job_name" and "job_id" in values and values["job_id"]:
            return None
        
        # If neither job_id nor job_name is set, we need a notebook_path or python_file_path
        if v is None:
            # This validation will happen during the model init since we need to check other fields
            pass
            
        return v
    
    @field_validator("notebook_path", "python_file_path")
    @classmethod
    def validate_notebook_or_file(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """
        Validate that if no job_id or job_name, either notebook_path or
        python_file_path must be provided.
        """
        # Skip this validation if job_id or job_name is provided
        if ("job_id" in values and values["job_id"]) or ("job_name" in values and values["job_name"]):
            return v
            
        # If validating notebook_path and python_file_path is already set
        if v is None and values["_field_name"] == "notebook_path" and "python_file_path" in values and values["python_file_path"]:
            return None
            
        # If validating python_file_path and notebook_path is already set    
        if v is None and values["_field_name"] == "python_file_path" and "notebook_path" in values and values["notebook_path"]:
            return None
            
        if v is None:
            # This final check should happen at model init
            pass
            
        return v
            
    def model_post_init(self, __context: Any) -> None:
        """Additional validation after model initialization."""
        if not self.job_id and not self.job_name and not self.notebook_path and not self.python_file_path:
            raise ValueError(
                "Either job_id, job_name, notebook_path, or python_file_path must be provided"
            )


class MLFlowModelVersionInfo(BaseModel):
    """Information about an MLflow model version with strong typing."""
    name: str
    version: str
    status: str
    current_stage: str
    creation_timestamp: int
    last_updated_timestamp: int
    description: Optional[str] = None
    user_id: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    run_id: Optional[str] = None
    run_link: Optional[str] = None
    source: Optional[str] = None


class JobRunResult(Result):
    """Result of a Databricks job run with strong typing."""
    run_id: Optional[str] = None
    run_url: Optional[str] = None
    job_id: Optional[str] = None
    job_name: Optional[str] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    setup_duration: Optional[int] = None
    execution_duration: Optional[int] = None
    cleanup_duration: Optional[int] = None
    trigger: Optional[str] = None
    creator_user_name: Optional[str] = None
    tasks: List[Dict[str, Any]] = Field(default_factory=list)


class DatabricksClient:
    """
    Strongly-typed client for interacting with Databricks resources.
    
    This class provides a typed interface to Databricks resources and handles
    authentication and error handling.
    """
    
    def __init__(self, config: DatabricksConfig):
        """
        Initialize the Databricks client.
        
        Args:
            config: Databricks connection configuration
        """
        self.config = config
        self.logger = logging.getLogger("DatabricksClient")
        self._client = self._create_client()
    
    def _create_client(self) -> WorkspaceClient:
        """
        Create and configure a Databricks SDK client.
        
        Returns:
            WorkspaceClient: Configured Databricks client
        """
        # Configuration parameters for the client
        kwargs = {}
        
        if self.config.host:
            kwargs["host"] = self.config.host
            
        if self.config.token:
            kwargs["token"] = self.config.token
            
        if self.config.profile:
            kwargs["profile"] = self.config.profile
        
        # Create the client with appropriate authentication
        try:
            return WorkspaceClient(**kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create Databricks client: {str(e)}")
            raise
    
    def run_job(self, params: JobRunParams) -> JobRunResult:
        """
        Run a Databricks job with strongly-typed parameters and results.
        
        Args:
            params: Job run parameters
            
        Returns:
            JobRunResult: The result of the job run
        """
        try:
            # Run an existing job if specified
            if params.job_id:
                response = self._client.jobs.run_now(
                    job_id=params.job_id,
                    jar_params=None,
                    notebook_params=params.parameters if params.notebook_path else None,
                    python_params=None,
                    spark_submit_params=None,
                    python_named_params=params.parameters if params.python_file_path else None,
                    idempotency_token=params.idempotency_token
                )
                run_id = response.run_id
                
            # Lookup job by name and run it
            elif params.job_name:
                # Find the job by name
                jobs_list = self._client.jobs.list()
                job_id = None
                
                for job in jobs_list:
                    if job.settings and job.settings.name == params.job_name:
                        job_id = job.job_id
                        break
                
                if not job_id:
                    return JobRunResult(
                        status=StatusEnum.FAILED,
                        message=f"Job with name '{params.job_name}' not found",
                        errors=[{"type": "JobNotFound", "message": f"No job found with name '{params.job_name}'"}]
                    )
                
                # Run the job
                response = self._client.jobs.run_now(
                    job_id=job_id,
                    jar_params=None,
                    notebook_params=params.parameters if params.notebook_path else None,
                    python_params=None,
                    spark_submit_params=None,
                    python_named_params=params.parameters if params.python_file_path else None,
                    idempotency_token=params.idempotency_token
                )
                run_id = response.run_id
                
            # Create and run a one-time job
            else:
                # Create a new task based on notebook or Python file
                task = None
                if params.notebook_path:
                    task = jobs.SubmitTask(
                        notebook_task=jobs.NotebookTask(
                            notebook_path=params.notebook_path,
                            base_parameters=params.parameters
                        )
                    )
                elif params.python_file_path:
                    task = jobs.SubmitTask(
                        spark_python_task=jobs.SparkPythonTask(
                            python_file=params.python_file_path,
                            parameters=[f"{k}={v}" for k, v in params.parameters.items()]
                        )
                    )
                
                # Define the run configuration
                run_config = jobs.SubmitRunRequest(
                    tasks=[task],
                    run_name=f"one-time-run-{params.notebook_path or params.python_file_path}",
                    timeout_seconds=params.timeout_seconds,
                    idempotency_token=params.idempotency_token
                )
                
                # If a specific cluster is specified
                if params.cluster_id:
                    run_config.existing_cluster_id = params.cluster_id
                
                # Submit the run
                response = self._client.jobs.submit(run_config)
                run_id = response.run_id
            
            # Get run details
            run_details = self._client.jobs.get_run(run_id=run_id)
            
            return JobRunResult(
                status=StatusEnum.SUCCESS,
                message=f"Job run submitted successfully with run_id: {run_id}",
                run_id=str(run_id),
                run_url=run_details.run_page_url,
                job_id=str(run_details.job_id) if run_details.job_id else None,
                job_name=run_details.run_name,
                start_time=run_details.start_time,
                trigger=run_details.trigger,
                creator_user_name=run_details.creator_user_name,
            )
            
        except Exception as e:
            self.logger.exception(f"Error running job: {str(e)}")
            return JobRunResult(
                status=StatusEnum.FAILED,
                message=f"Failed to run job: {str(e)}",
                errors=[{"type": type(e).__name__, "message": str(e)}]
            )
    
    def get_mlflow_model_versions(self, model_name: str) -> List[MLFlowModelVersionInfo]:
        """
        Get information about all versions of an MLflow model.
        
        Args:
            model_name: The name of the MLflow model
            
        Returns:
            List of MLFlowModelVersionInfo objects for each version
        """
        try:
            # Get the model versions
            model_versions = self._client.mlflow.search_model_versions(f"name='{model_name}'")
            
            # Convert to strongly-typed objects
            result = []
            for version in model_versions:
                model_version_info = MLFlowModelVersionInfo(
                    name=version.name,
                    version=version.version,
                    status=version.status,
                    current_stage=version.current_stage,
                    creation_timestamp=version.creation_timestamp,
                    last_updated_timestamp=version.last_updated_timestamp,
                    description=version.description,
                    user_id=version.user_id,
                    tags={t.key: t.value for t in (version.tags or [])},
                    run_id=version.run_id,
                    run_link=None,  # Not available in the API response
                    source=version.source,
                )
                result.append(model_version_info)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Error getting MLflow model versions: {str(e)}")
            return []
    
    def transition_model_version_stage(
        self, model_name: str, version: str, stage: str
    ) -> Result:
        """
        Transition an MLflow model version to a different stage.
        
        Args:
            model_name: The name of the MLflow model
            version: The version of the model
            stage: The target stage (e.g., "Staging", "Production")
            
        Returns:
            Result: The result of the transition operation
        """
        try:
            # Transition the model version
            self._client.mlflow.transition_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True  # Archive existing versions in the target stage
            )
            
            return Result(
                status=StatusEnum.SUCCESS,
                message=f"Model {model_name} version {version} transitioned to {stage} successfully",
                details={
                    "model_name": model_name,
                    "version": version,
                    "stage": stage
                }
            )
            
        except Exception as e:
            self.logger.exception(f"Error transitioning model version stage: {str(e)}")
            return Result(
                status=StatusEnum.FAILED,
                message=f"Failed to transition model version stage: {str(e)}",
                errors=[{"type": type(e).__name__, "message": str(e)}]
            )
