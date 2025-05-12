"""
Configuration models for Databricks MLOps framework.
"""
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from databricks_mlops.models.base import EnvironmentEnum, ValidationSeverity


class DatabaseConfig(BaseModel):
    """Configuration for database connections."""
    catalog: str
    schema: str
    warehouse_id: Optional[str] = None
    connection_parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 120


class DataValidationRule(BaseModel):
    """Configuration for a single data validation rule."""
    name: str
    condition: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    description: Optional[str] = None
    
    @field_validator("condition")
    @classmethod
    def validate_condition(cls, v: str) -> str:
        """Validate that the condition is not empty."""
        if not v.strip():
            raise ValueError("Condition cannot be empty")
        return v


class DataConfig(BaseModel):
    """Configuration for data pipelines."""
    source_path: str
    destination_path: str
    table_name: str
    format: str = "delta"
    partition_columns: List[str] = Field(default_factory=list)
    validation_rules: List[Union[str, DataValidationRule]] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)
    version: Optional[str] = None
    database_config: Optional[DatabaseConfig] = None
    
    @field_validator("validation_rules")
    @classmethod
    def convert_str_to_rule(cls, rules: List[Union[str, DataValidationRule]]) -> List[DataValidationRule]:
        """Convert string validation rules to DataValidationRule objects."""
        result = []
        for rule in rules:
            if isinstance(rule, str):
                result.append(DataValidationRule(
                    name=f"rule_{len(result)}",
                    condition=rule
                ))
            else:
                result.append(rule)
        return result


class FeatureConfig(BaseModel):
    """Configuration for feature engineering pipelines."""
    source_table: str
    feature_table_name: str
    primary_keys: List[str]
    features: List[str]
    timestamp_column: Optional[str] = None
    partition_columns: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    online_store_enabled: bool = False
    offline_store_path: Optional[str] = None


class ModelConfig(BaseModel):
    """Configuration for model training and deployment."""
    model_name: str
    model_type: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    features: List[str]
    target_column: str
    train_data_path: str
    validation_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    metrics: List[str] = Field(default_factory=list)
    autologging_enabled: bool = True
    register_model: bool = True
    save_artifacts: bool = True
    max_parallel_trials: int = 1
    timeout_minutes: int = 60


class DeploymentConfig(BaseModel):
    """Configuration for model deployment."""
    model_name: str
    model_version: Optional[str] = None
    environment: EnvironmentEnum
    deployment_type: str = "serving_endpoint"  # or "batch_inference", "streaming_inference"
    endpoint_name: Optional[str] = None
    compute_type: str = "cpu"  # or "gpu"
    compute_scale: int = 1
    min_replicas: int = 1
    max_replicas: int = 2
    autoscaling_enabled: bool = True
    enable_access_control: bool = True
    timeout_seconds: int = 300
    tags: Dict[str, str] = Field(default_factory=dict)
    environment_variables: Dict[str, str] = Field(default_factory=dict)


class MonitoringConfig(BaseModel):
    """Configuration for model monitoring."""
    model_name: str
    endpoint_name: Optional[str] = None
    metrics: List[str] = Field(default_factory=list)
    monitor_data_drift: bool = True
    monitor_prediction_drift: bool = True
    reference_dataset_path: Optional[str] = None
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)
    monitoring_schedule: str = "0 0 * * *"  # Default: daily at midnight (CRON format)
    lookback_days: int = 7
    alert_emails: List[str] = Field(default_factory=list)
    dashboard_name: Optional[str] = None


class PipelineConfig(BaseModel):
    """Base configuration for all pipeline types."""
    name: str
    description: Optional[str] = None
    owner: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    timeout_minutes: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: int = 60
    environment: EnvironmentEnum = EnvironmentEnum.DEV
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Union of the specific configs
    data_config: Optional[DataConfig] = None
    feature_config: Optional[FeatureConfig] = None
    model_config: Optional[ModelConfig] = None
    deployment_config: Optional[DeploymentConfig] = None
    monitoring_config: Optional[MonitoringConfig] = None
