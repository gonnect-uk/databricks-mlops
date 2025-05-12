"""
Core pipeline abstractions for the Databricks MLOps framework.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel

from databricks_mlops.models.base import Result, StatusEnum, ValidationResult
from databricks_mlops.models.config import PipelineConfig

# Type variables for generic pipeline implementations
T = TypeVar('T', bound=PipelineConfig)
R = TypeVar('R', bound=Result)

logger = logging.getLogger(__name__)


class Pipeline(Generic[T, R], ABC):
    """
    Abstract base class for all MLOps pipelines.
    
    This class defines the common interface and behavior for all pipeline types
    in the Databricks MLOps framework.
    """
    
    def __init__(self, config: T):
        """Initialize the pipeline with a configuration."""
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{self.config.name}")
    
    @abstractmethod
    def validate(self) -> ValidationResult:
        """
        Validate the pipeline configuration and prerequisites.
        
        Returns:
            ValidationResult: The result of the validation.
        """
        pass
    
    @abstractmethod
    def run(self) -> R:
        """
        Execute the pipeline.
        
        Returns:
            R: The result of the pipeline execution.
        """
        pass
    
    def before_run(self) -> None:
        """Hook executed before the pipeline runs."""
        self.logger.info(f"Starting pipeline: {self.config.name}")
    
    def after_run(self, result: R) -> None:
        """
        Hook executed after the pipeline runs.
        
        Args:
            result (R): The result of the pipeline execution.
        """
        status_message = "succeeded" if result.is_success else "failed"
        self.logger.info(f"Pipeline {self.config.name} {status_message} with message: {result.message}")
    
    def handle_error(self, error: Exception) -> R:
        """
        Handle errors that occur during pipeline execution.
        
        Args:
            error (Exception): The exception that was raised.
            
        Returns:
            R: A result object representing the error.
        """
        self.logger.exception(f"Error in pipeline {self.config.name}: {str(error)}")
        # This method should be implemented by subclasses to return the appropriate result type
        raise NotImplementedError("Subclasses must implement handle_error")


class DataPipeline(Pipeline[PipelineConfig, Result]):
    """
    Pipeline for data ingestion, validation, and transformation.
    """
    
    def validate(self) -> ValidationResult:
        """
        Validate the data pipeline configuration and source data.
        
        Returns:
            ValidationResult: The result of the validation.
        """
        if self.config.data_config is None:
            return ValidationResult(
                status=StatusEnum.FAILED,
                message="Data configuration is required for DataPipeline",
                validation_type="config",
                failed_validations=[{
                    "rule": "data_config_present",
                    "message": "DataPipeline requires data_config to be defined"
                }]
            )
        
        # Validate source path exists
        # Validate destination permissions
        # Validate table name format
        # Implement more detailed validation logic
        
        return ValidationResult(
            status=StatusEnum.SUCCESS,
            message="Data pipeline configuration validated successfully",
            validation_type="config",
            validation_rules=["data_config_present"]
        )
    
    def run(self) -> Result:
        """
        Execute the data pipeline with ingestion, validation, and transformation steps.
        
        Returns:
            Result: The result of the pipeline execution.
        """
        try:
            self.before_run()
            
            validation_result = self.validate()
            if not validation_result.is_success:
                return Result(
                    status=StatusEnum.FAILED,
                    message=f"Validation failed: {validation_result.message}",
                    details={"validation_result": validation_result.dict()}
                )
            
            # Implement the actual data pipeline logic here
            # This would include steps like:
            # - Loading data from source
            # - Validating data quality
            # - Transforming data
            # - Writing to destination
            
            result = Result(
                status=StatusEnum.SUCCESS,
                message=f"Data pipeline {self.config.name} completed successfully",
                details={
                    "source_path": self.config.data_config.source_path,
                    "destination_path": self.config.data_config.destination_path,
                    "table_name": self.config.data_config.table_name
                }
            )
            
            self.after_run(result)
            return result
            
        except Exception as e:
            return self.handle_error(e)
    
    def handle_error(self, error: Exception) -> Result:
        """
        Handle errors in the data pipeline.
        
        Args:
            error (Exception): The exception that was raised.
            
        Returns:
            Result: A result object representing the error.
        """
        self.logger.exception(f"Data pipeline error: {str(error)}")
        return Result(
            status=StatusEnum.FAILED,
            message=f"Data pipeline failed: {str(error)}",
            errors=[{"type": type(error).__name__, "message": str(error)}]
        )


class FeaturePipeline(Pipeline[PipelineConfig, Result]):
    """
    Pipeline for feature engineering and feature store integration.
    """
    
    def validate(self) -> ValidationResult:
        """
        Validate feature pipeline configuration.
        
        Returns:
            ValidationResult: The result of the validation.
        """
        if self.config.feature_config is None:
            return ValidationResult(
                status=StatusEnum.FAILED,
                message="Feature configuration is required for FeaturePipeline",
                validation_type="config",
                failed_validations=[{
                    "rule": "feature_config_present",
                    "message": "FeaturePipeline requires feature_config to be defined"
                }]
            )
        
        # Add detailed validation logic
        
        return ValidationResult(
            status=StatusEnum.SUCCESS,
            message="Feature pipeline configuration validated successfully",
            validation_type="config",
            validation_rules=["feature_config_present"]
        )
    
    def run(self) -> Result:
        """
        Execute the feature engineering pipeline.
        
        Returns:
            Result: The result of the pipeline execution.
        """
        try:
            self.before_run()
            
            validation_result = self.validate()
            if not validation_result.is_success:
                return Result(
                    status=StatusEnum.FAILED,
                    message=f"Validation failed: {validation_result.message}",
                    details={"validation_result": validation_result.dict()}
                )
            
            # Implement feature engineering logic
            # - Extract features from source table
            # - Transform features
            # - Register to feature store
            
            result = Result(
                status=StatusEnum.SUCCESS,
                message=f"Feature pipeline {self.config.name} completed successfully",
                details={
                    "source_table": self.config.feature_config.source_table,
                    "feature_table": self.config.feature_config.feature_table_name,
                    "features": self.config.feature_config.features
                }
            )
            
            self.after_run(result)
            return result
            
        except Exception as e:
            return self.handle_error(e)
    
    def handle_error(self, error: Exception) -> Result:
        """Handle errors in the feature pipeline."""
        self.logger.exception(f"Feature pipeline error: {str(error)}")
        return Result(
            status=StatusEnum.FAILED,
            message=f"Feature pipeline failed: {str(error)}",
            errors=[{"type": type(error).__name__, "message": str(error)}]
        )


class TrainingPipeline(Pipeline[PipelineConfig, Result]):
    """
    Pipeline for model training, evaluation, and registration.
    """
    
    def validate(self) -> ValidationResult:
        """
        Validate model training configuration.
        
        Returns:
            ValidationResult: The result of the validation.
        """
        if self.config.model_config is None:
            return ValidationResult(
                status=StatusEnum.FAILED,
                message="Model configuration is required for TrainingPipeline",
                validation_type="config",
                failed_validations=[{
                    "rule": "model_config_present",
                    "message": "TrainingPipeline requires model_config to be defined"
                }]
            )
        
        # Add detailed validation logic
        
        return ValidationResult(
            status=StatusEnum.SUCCESS,
            message="Training pipeline configuration validated successfully",
            validation_type="config",
            validation_rules=["model_config_present"]
        )
    
    def run(self) -> Result:
        """
        Execute the model training pipeline.
        
        Returns:
            Result: The result of the pipeline execution.
        """
        try:
            self.before_run()
            
            validation_result = self.validate()
            if not validation_result.is_success:
                return Result(
                    status=StatusEnum.FAILED,
                    message=f"Validation failed: {validation_result.message}",
                    details={"validation_result": validation_result.dict()}
                )
            
            # Implement model training logic
            # - Prepare training data
            # - Train model
            # - Evaluate model
            # - Register model to MLflow
            
            result = Result(
                status=StatusEnum.SUCCESS,
                message=f"Training pipeline {self.config.name} completed successfully",
                details={
                    "model_name": self.config.model_config.model_name,
                    "model_type": self.config.model_config.model_type,
                    "metrics": {"accuracy": 0.92, "f1_score": 0.91}  # Example metrics
                }
            )
            
            self.after_run(result)
            return result
            
        except Exception as e:
            return self.handle_error(e)
    
    def handle_error(self, error: Exception) -> Result:
        """Handle errors in the training pipeline."""
        self.logger.exception(f"Training pipeline error: {str(error)}")
        return Result(
            status=StatusEnum.FAILED,
            message=f"Training pipeline failed: {str(error)}",
            errors=[{"type": type(error).__name__, "message": str(error)}]
        )


class DeploymentPipeline(Pipeline[PipelineConfig, Result]):
    """
    Pipeline for model deployment to different environments.
    """
    
    def validate(self) -> ValidationResult:
        """
        Validate model deployment configuration.
        
        Returns:
            ValidationResult: The result of the validation.
        """
        if self.config.deployment_config is None:
            return ValidationResult(
                status=StatusEnum.FAILED,
                message="Deployment configuration is required for DeploymentPipeline",
                validation_type="config",
                failed_validations=[{
                    "rule": "deployment_config_present",
                    "message": "DeploymentPipeline requires deployment_config to be defined"
                }]
            )
        
        # Add detailed validation logic
        
        return ValidationResult(
            status=StatusEnum.SUCCESS,
            message="Deployment pipeline configuration validated successfully",
            validation_type="config",
            validation_rules=["deployment_config_present"]
        )
    
    def run(self) -> Result:
        """
        Execute the model deployment pipeline.
        
        Returns:
            Result: The result of the pipeline execution.
        """
        try:
            self.before_run()
            
            validation_result = self.validate()
            if not validation_result.is_success:
                return Result(
                    status=StatusEnum.FAILED,
                    message=f"Validation failed: {validation_result.message}",
                    details={"validation_result": validation_result.dict()}
                )
            
            # Implement model deployment logic
            # - Get model from registry
            # - Create/update serving endpoint
            # - Verify deployment
            # - Update tags and documentation
            
            result = Result(
                status=StatusEnum.SUCCESS,
                message=f"Deployment pipeline {self.config.name} completed successfully",
                details={
                    "model_name": self.config.deployment_config.model_name,
                    "environment": self.config.deployment_config.environment,
                    "endpoint_name": self.config.deployment_config.endpoint_name or f"{self.config.deployment_config.model_name}-endpoint"
                }
            )
            
            self.after_run(result)
            return result
            
        except Exception as e:
            return self.handle_error(e)
    
    def handle_error(self, error: Exception) -> Result:
        """Handle errors in the deployment pipeline."""
        self.logger.exception(f"Deployment pipeline error: {str(error)}")
        return Result(
            status=StatusEnum.FAILED,
            message=f"Deployment pipeline failed: {str(error)}",
            errors=[{"type": type(error).__name__, "message": str(error)}]
        )


class MonitoringPipeline(Pipeline[PipelineConfig, Result]):
    """
    Pipeline for model monitoring and drift detection.
    """
    
    def validate(self) -> ValidationResult:
        """
        Validate model monitoring configuration.
        
        Returns:
            ValidationResult: The result of the validation.
        """
        if self.config.monitoring_config is None:
            return ValidationResult(
                status=StatusEnum.FAILED,
                message="Monitoring configuration is required for MonitoringPipeline",
                validation_type="config",
                failed_validations=[{
                    "rule": "monitoring_config_present",
                    "message": "MonitoringPipeline requires monitoring_config to be defined"
                }]
            )
        
        # Add detailed validation logic
        
        return ValidationResult(
            status=StatusEnum.SUCCESS,
            message="Monitoring pipeline configuration validated successfully",
            validation_type="config",
            validation_rules=["monitoring_config_present"]
        )
    
    def run(self) -> Result:
        """
        Execute the model monitoring pipeline.
        
        Returns:
            Result: The result of the pipeline execution.
        """
        try:
            self.before_run()
            
            validation_result = self.validate()
            if not validation_result.is_success:
                return Result(
                    status=StatusEnum.FAILED,
                    message=f"Validation failed: {validation_result.message}",
                    details={"validation_result": validation_result.dict()}
                )
            
            # Implement model monitoring logic
            # - Collect metrics
            # - Detect data drift
            # - Generate alerts if needed
            # - Update dashboards
            
            result = Result(
                status=StatusEnum.SUCCESS,
                message=f"Monitoring pipeline {self.config.name} completed successfully",
                details={
                    "model_name": self.config.monitoring_config.model_name,
                    "drift_detected": False,  # Example
                    "metrics": {
                        "prediction_drift": 0.02,
                        "data_drift": 0.03
                    }
                }
            )
            
            self.after_run(result)
            return result
            
        except Exception as e:
            return self.handle_error(e)
    
    def handle_error(self, error: Exception) -> Result:
        """Handle errors in the monitoring pipeline."""
        self.logger.exception(f"Monitoring pipeline error: {str(error)}")
        return Result(
            status=StatusEnum.FAILED,
            message=f"Monitoring pipeline failed: {str(error)}",
            errors=[{"type": type(error).__name__, "message": str(error)}]
        )
