# Databricks MLOps Framework API Reference

This document provides a comprehensive reference for the strongly-typed components of the Databricks MLOps framework.

## Installation and Package Extras

The framework is optimized for installation with [uv](https://github.com/astral-sh/uv), a modern Python package installer that offers significant speed and reliability improvements:

```bash
# Install core framework
uv pip install databricks-mlops

# Install with specific extras based on your needs
uv pip install "databricks-mlops[ml]"         # Feature engineering and model training components
uv pip install "databricks-mlops[databricks]"  # Databricks integration
uv pip install "databricks-mlops[production]"  # Monitoring and deployment components
uv pip install "databricks-mlops[all]"         # All components
```

Available extras:

| Extra | Description |
|-------|-------------|
| `data` | Data validation components using Great Expectations |
| `feature-engineering` | Feature transformation and encoding |
| `model-training` | Model training with MLflow integration |
| `drift-detection` | Statistical drift detection capabilities |
| `databricks` | Databricks integration components |
| `api` | FastAPI-based model serving components |
| `monitoring` | Prometheus metrics and Grafana dashboards |
| `deployment` | Deployment to Databricks endpoints |
| `ml` | Combined ML extras (feature-engineering, model-training, drift-detection) |
| `production` | Combined production extras (monitoring, deployment, databricks, api) |
| `all` | All components |

These extras follow our strict type-safe philosophy, ensuring that each component maintains strong typing and Pydantic models throughout.

## Table of Contents

- [Core Components](#core-components)
  - [Pipeline Base Classes](#pipeline-base-classes)
  - [Pipeline Orchestrator](#pipeline-orchestrator)
- [Models](#models)
  - [Base Models](#base-models)
  - [Configuration Models](#configuration-models)
- [Pipelines](#pipelines)
  - [Feature Engineering](#feature-engineering)
  - [Model Training](#model-training)
  - [Model Deployment](#model-deployment)
- [Monitoring](#monitoring)
  - [Drift Detection](#drift-detection)
  - [Metric Collection](#metric-collection)
- [Utilities](#utilities)
  - [Data Validation](#data-validation)
  - [Configuration Management](#configuration-management)
  - [Logging](#logging)
  - [Databricks Client](#databricks-client)

## Core Components

### Pipeline Base Classes

All pipeline classes inherit from abstract base classes to ensure consistent behavior and interfaces.

#### `Pipeline`

```python
class Pipeline(ABC):
    """
    Abstract base class for all pipelines with common execution patterns.
    """
    name: str
    logger: Logger
    
    @abstractmethod
    def run(self, **kwargs) -> Result:
        """Execute the pipeline and return the result."""
        pass
    
    @abstractmethod
    def validate(self) -> ValidationResult:
        """Validate pipeline configuration and requirements."""
        pass
```

#### `DataPipeline`

```python
class DataPipeline(Pipeline):
    """
    Pipeline for data processing and validation operations.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.name = config.name
        self.logger = setup_logger(self.name)
        self._validate_config()
    
    def run(self, **kwargs) -> Result:
        """Execute the data pipeline."""
        pass
        
    def validate(self) -> ValidationResult:
        """Validate data pipeline configuration."""
        pass
```

Similar class definitions exist for `FeaturePipeline`, `TrainingPipeline`, `DeploymentPipeline`, and `MonitoringPipeline`.

### Pipeline Orchestrator

The `PipelineOrchestrator` manages the execution of multiple pipeline stages with dependencies.

```python
class PipelineOrchestrator:
    """
    Orchestrates multiple pipeline stages with dependency management.
    """
    
    def __init__(self):
        self.stages: List[PipelineStage] = []
        self.logger = setup_logger("orchestrator")
    
    def add_data_stage(self, name: str, config: PipelineConfig, 
                      depends_on: Optional[List[str]] = None, 
                      enabled: bool = True) -> None:
        """Add a data processing stage to the orchestration."""
        pass
        
    # Similar methods for other pipeline types
    
    def run(self) -> Result:
        """Run the orchestrated pipeline respecting dependencies."""
        pass
```

## Models

### Base Models

The framework uses Pydantic models for all data structures to ensure type safety.

#### `Result`

```python
class Result(BaseModel):
    """Base result model for all operations."""
    status: StatusEnum
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Dict[str, Any] = Field(default_factory=dict)
```

#### `ValidationResult`

```python
class ValidationResult(Result):
    """Result of validation operations."""
    validation_errors: List[ValidationError] = Field(default_factory=list)
    failed_rules: List[str] = Field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Return True if validation passed."""
        return self.status == StatusEnum.SUCCESS
```

#### `DriftResult`

```python
class DriftResult(Result):
    """Result of drift detection operations."""
    has_drift: bool = False
    drift_score: float = 0.0
    drifted_features: List[str] = Field(default_factory=list)
    feature_drift_scores: Dict[str, float] = Field(default_factory=dict)
```

### Configuration Models

Configuration models provide strong typing for all configuration options.

#### `PipelineConfig`

```python
class PipelineConfig(BaseModel):
    """Base configuration for all pipelines."""
    name: str
    description: str = ""
    owner: str = ""
    tags: Dict[str, str] = Field(default_factory=dict)
    timeout_minutes: int = 60
    retry_attempts: int = 0
    environment: str = "development"
    
    # Optional configurations for specific pipeline types
    data_config: Optional[DataConfig] = None
    feature_config: Optional[FeatureConfig] = None
    model_config: Optional[ModelConfig] = None
    deployment_config: Optional[DeploymentConfig] = None
    monitoring_config: Optional[MonitoringConfig] = None
```

## Pipelines

### Feature Engineering

The feature engineering pipeline handles data transformations with strong typing.

#### `FeatureTransformer`

```python
class FeatureTransformer:
    """
    Handles feature transformations with strong typing.
    """
    
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.transformers: Dict[str, Any] = {}
        self.logger = setup_logger("feature_transformer")
        self._fitted = False
        
    def fit(self, data: pd.DataFrame) -> 'FeatureTransformer':
        """Fit transformers to the data."""
        pass
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted transformers."""
        pass
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit transformers and transform data."""
        pass
```

### Model Training

The model training pipeline handles model training, evaluation, and registration.

#### `ModelTrainer`

```python
class ModelTrainer:
    """
    Handles model training, evaluation, and registration.
    """
    
    def __init__(self, config: TrainingConfig, tracking_config: Optional[TrackingConfig] = None):
        self.config = config
        self.tracking_config = tracking_config or TrackingConfig()
        self.logger = setup_logger("model_trainer")
        
    def train_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: Optional[pd.DataFrame] = None, 
                           y_val: Optional[pd.Series] = None,
                           register_model: bool = True) -> Result:
        """Train, evaluate, and optionally register a model."""
        pass
```

### Model Deployment

The model deployment pipeline handles deploying models to various environments.

#### `ModelDeployer`

```python
class ModelDeployer:
    """
    Handles model deployment to various environments.
    """
    
    def __init__(self, config: ModelDeploymentConfig):
        self.config = config
        self.logger = setup_logger("model_deployer")
        
    def deploy_to_endpoint(self, model_uri: str) -> Result:
        """Deploy a model to a serving endpoint."""
        pass
        
    def deploy_to_batch(self, model_uri: str) -> Result:
        """Deploy a model for batch inference."""
        pass
```

## Monitoring

### Drift Detection

The drift detection system monitors for changes in data and model behavior.

#### `DriftDetector`

```python
class DriftDetector:
    """
    Detects data and concept drift using statistical methods.
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = setup_logger("drift_detector")
        
    def detect_drift(self, reference_data: pd.DataFrame, 
                    current_data: pd.DataFrame,
                    categorical_features: Optional[List[str]] = None,
                    numerical_features: Optional[List[str]] = None) -> DriftResult:
        """
        Detect drift between reference and current data.
        """
        pass
```

### Metric Collection

The metric collection system gathers and analyzes model performance metrics.

#### `MetricCollector`

```python
class MetricCollector:
    """
    Collects and analyzes model performance metrics.
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = setup_logger("metric_collector")
        
    def collect_metrics(self, y_true: pd.Series, y_pred: pd.Series,
                        timestamps: Optional[pd.Series] = None) -> Result:
        """
        Collect and compute performance metrics.
        """
        pass
```

## Utilities

### Data Validation

The data validation utilities ensure data quality.

#### `DataValidator`

```python
class DataValidator:
    """
    Validates data quality using predefined rules.
    """
    
    def __init__(self, rules: List[ValidationRule]):
        self.rules = rules
        self.logger = setup_logger("data_validator")
        
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate data against rules.
        """
        pass
```

### Configuration Management

The configuration management system handles loading and validating configurations.

#### `MLOpsConfigManager`

```python
class MLOpsConfigManager:
    """
    Manages loading and validating configurations.
    """
    
    @classmethod
    def create_pipeline_config_manager(cls) -> 'MLOpsConfigManager':
        """Create a configuration manager for pipelines."""
        pass
        
    def load_from_yaml(self, file_path: str) -> PipelineConfig:
        """Load configuration from YAML file."""
        pass
        
    def load_from_dict(self, config_dict: Dict[str, Any]) -> PipelineConfig:
        """Load configuration from dictionary."""
        pass
```

### Logging

The logging utilities provide structured logging.

#### `setup_logger`

```python
def setup_logger(name: str, level: LogLevel = LogLevel.INFO) -> Logger:
    """
    Set up a structured logger.
    """
    pass
```

### Databricks Client

The Databricks client provides strongly-typed interfaces to Databricks APIs.

#### `DatabricksClient`

```python
class DatabricksClient:
    """
    Client for interacting with Databricks APIs.
    """
    
    def __init__(self, config: DatabricksConfig):
        self.config = config
        self.logger = setup_logger("databricks_client")
        
    def get_model_version(self, model_name: str, version: str) -> ModelVersion:
        """
        Get a specific model version.
        """
        pass
        
    def create_job_run(self, job_id: str, params: Dict[str, Any]) -> JobRun:
        """
        Create a job run.
        """
        pass
```
