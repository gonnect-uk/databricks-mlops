"""
Base model definitions for the Databricks MLOps framework.
"""
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class StatusEnum(str, Enum):
    """Status enum for tracking the status of pipeline runs."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    WARNING = "warning"
    CANCELLED = "cancelled"


class EnvironmentEnum(str, Enum):
    """Environment enum for defining deployment targets."""
    DEV = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ResourceType(str, Enum):
    """Resource types in the Databricks ecosystem."""
    NOTEBOOK = "notebook"
    DBFS_FILE = "dbfs_file"
    CLUSTER = "cluster"
    JOB = "job"
    MLFLOW_MODEL = "mlflow_model"
    FEATURE_TABLE = "feature_table"
    DELTA_TABLE = "delta_table"
    DASHBOARD = "dashboard"
    SECRET = "secret"
    SERVICE_PRINCIPAL = "service_principal"


class ValidationSeverity(str, Enum):
    """Severity levels for data validation errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class BaseResource(BaseModel):
    """Base class for all resource models in the framework."""
    name: str
    resource_type: ResourceType
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    owner: Optional[str] = None
    description: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        frozen = False
        validate_assignment = True
        extra = "forbid"


class Result(BaseModel):
    """Base result model for operation outcomes."""
    status: StatusEnum
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    
    @property
    def is_success(self) -> bool:
        """Check if the result status indicates success."""
        return self.status == StatusEnum.SUCCESS
    
    @property
    def is_failed(self) -> bool:
        """Check if the result status indicates failure."""
        return self.status == StatusEnum.FAILED
    
    @property
    def has_warnings(self) -> bool:
        """Check if the result has warnings."""
        return self.status == StatusEnum.WARNING or len(self.errors) > 0


class ValidationResult(Result):
    """Result model specifically for validation operations."""
    validation_type: str
    validation_rules: List[str] = Field(default_factory=list)
    passed_validations: List[str] = Field(default_factory=list)
    failed_validations: List[Dict[str, Any]] = Field(default_factory=list)
    severity: ValidationSeverity = ValidationSeverity.ERROR
    
    @property
    def all_passed(self) -> bool:
        """Check if all validations passed."""
        return len(self.failed_validations) == 0
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of validations."""
        total = len(self.passed_validations) + len(self.failed_validations)
        if total == 0:
            return 1.0
        return len(self.passed_validations) / total


class Metric(BaseModel):
    """Model for tracking metrics."""
    name: str
    value: Union[float, int, str, bool]
    timestamp: datetime = Field(default_factory=datetime.now)
    labels: Dict[str, str] = Field(default_factory=dict)
    
    @field_validator("value")
    @classmethod
    def validate_metric_value(cls, v: Any) -> Any:
        """Validate that the metric value is of supported types."""
        if not isinstance(v, (float, int, str, bool)):
            raise ValueError(f"Metric value must be float, int, str, or bool, got {type(v)}")
        return v


class LogRecord(BaseModel):
    """Model for structured logging."""
    level: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    context: Dict[str, Any] = Field(default_factory=dict)
    logger: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


class ArtifactMetadata(BaseModel):
    """Metadata for tracked artifacts."""
    name: str
    artifact_type: str
    path: str
    size_bytes: Optional[int] = None
    md5_hash: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    version: Optional[str] = None
    stage: Optional[str] = None
    description: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
