# Databricks MLOps Framework Usage Guide

This guide provides practical examples of using the MLOps framework for Databricks, illustrating the type-safe, Pydantic-driven approach that powers the entire system.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Examples](#configuration-examples)
- [Data Processing](#data-processing)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Model Deployment](#model-deployment)
- [Model Monitoring](#model-monitoring)
- [Complete MLOps Pipeline](#complete-mlops-pipeline)
- [Best Practices](#best-practices)

## Installation

We recommend installing the framework using [uv](https://github.com/astral-sh/uv), a modern Python package installer that's significantly faster and more reliable than traditional pip:

```bash
# Install uv if you don't have it yet
curl -sSf https://astral.sh/uv/install.sh | bash

# Install from PyPI
uv pip install databricks-mlops

# Or install directly from the repository
uv pip install git+https://github.com/gonnect-uk/databricks-mlops.git

# Install with specific extras
uv pip install "databricks-mlops[monitoring,deployment]"

# Install all extras
uv pip install "databricks-mlops[all]"
```

If you prefer using traditional pip:

```bash
pip install databricks-mlops
```

## Quick Start

Here's a minimal example to get started:

```python
from databricks_mlops.config import MLOpsConfigManager
from databricks_mlops.pipelines import FeatureTransformer, ModelTrainer
import pandas as pd

# Load configurations with strong typing
config_manager = MLOpsConfigManager.create_pipeline_config_manager()
feature_config = config_manager.load_from_yaml("feature_config.yaml")
model_config = config_manager.load_from_yaml("model_config.yaml")

# Load data
data = pd.read_parquet("path/to/data.parquet")

# Feature engineering with type safety
transformer = FeatureTransformer(config=feature_config.feature_config)
transformed_data = transformer.fit_transform(data)

# Split data
X = transformed_data.drop('target', axis=1)
y = transformed_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
trainer = ModelTrainer(config=model_config.model_config)
result = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

print(f"Model training status: {result.status}")
print(f"Metrics: {result.metrics}")
```

## Configuration Examples

### Data Pipeline Configuration

```yaml
name: customer_churn_data_pipeline
description: "Data pipeline for customer churn prediction model"
owner: "data_engineer@example.com"
tags:
  domain: customer_analytics
  project: churn_prediction
  environment: development
timeout_minutes: 60
retry_attempts: 3

data_config:
  source_path: "dbfs:/mnt/raw-data/customer_data.parquet"
  destination_path: "dbfs:/mnt/silver/customer_data_processed"
  validation_rules:
    - name: "no_missing_ids"
      condition: "customer_id is not null"
      severity: "error"
    - name: "positive_tenure"
      condition: "tenure >= 0"
      severity: "error"
```

### Feature Engineering Configuration

```yaml
name: customer_churn_feature_pipeline
description: "Feature engineering pipeline for customer churn prediction model"
owner: "data_scientist@example.com"

feature_config:
  source_table: "main.analytics.customer_data_processed"
  feature_table_name: "customer_churn_features"
  primary_keys: 
    - "customer_id"
  features:
    - "tenure"
    - "monthly_charges"
    - "total_charges"
    - "contract_type"
  transformers:
    - name: "numeric_scaler"
      type: "standard_scaler"
      features: ["tenure", "monthly_charges", "total_charges"]
      scope: "numerical"
    - name: "categorical_encoder"
      type: "one_hot"
      features: ["contract_type", "payment_method"]
      scope: "categorical"
```

## Data Processing

Process and validate data with strong typing:

```python
from databricks_mlops.models.config import DataConfig
from databricks_mlops.utils.data_validation import DataValidator, ValidationRule
from databricks_mlops.core import DataPipeline
import pandas as pd

# Define validation rules with strong typing
rules = [
    ValidationRule(
        name="no_missing_ids",
        condition="customer_id is not null",
        severity="error",
        description="Customer ID should never be null"
    ),
    ValidationRule(
        name="positive_tenure",
        condition="tenure >= 0",
        severity="error",
        description="Tenure cannot be negative"
    )
]

# Create data validator
validator = DataValidator(rules=rules)

# Load data
data = pd.read_parquet("path/to/data.parquet")

# Validate data
validation_result = validator.validate(data)

if validation_result.is_valid:
    print("Data validation passed!")
else:
    print("Data validation failed!")
    for error in validation_result.failed_rules:
        print(f"- {error.name}: {error.description}")
```

## Feature Engineering

Transform features with type safety:

```python
from databricks_mlops.models.config import FeatureConfig
from databricks_mlops.pipelines import (FeatureTransformer, ScalerType, 
                                      FeatureScope, TransformerType)
import pandas as pd

# Define feature engineering configuration with Pydantic
feature_config = FeatureConfig(
    categorical_features=["contract_type", "payment_method"],
    numerical_features=["tenure", "monthly_charges", "total_charges"],
    target_column="churned",
    transformers=[
        {
            "name": "numeric_scaler",
            "type": ScalerType.STANDARD,
            "features": ["tenure", "monthly_charges", "total_charges"],
            "scope": FeatureScope.NUMERICAL
        },
        {
            "name": "categorical_encoder",
            "type": TransformerType.ENCODER,
            "encoder_type": "one_hot",
            "features": ["contract_type", "payment_method"],
            "scope": FeatureScope.CATEGORICAL
        }
    ]
)

# Create and use transformer
transformer = FeatureTransformer(config=feature_config)
data = pd.read_parquet("path/to/data.parquet")
transformed_data = transformer.fit_transform(data)

# Save transformer for later use
transformer.save("path/to/transformer.pkl")

# Later, load and use it
loaded_transformer = FeatureTransformer.load("path/to/transformer.pkl")
new_data = pd.read_parquet("path/to/new_data.parquet")
new_transformed = loaded_transformer.transform(new_data)
```

## Model Training

Train and evaluate models with proper metrics tracking:

```python
from databricks_mlops.models.config import ModelConfig
from databricks_mlops.pipelines import (ModelTrainer, ModelType, 
                                      ModelFramework, TrainingConfig)
from databricks_mlops.workflows.mlflow_tracking import TrackingConfig
import pandas as pd
from sklearn.model_selection import train_test_split

# Define training configuration using Pydantic
training_config = TrainingConfig(
    model_name="customer_churn_predictor",
    model_type=ModelType.CLASSIFICATION,
    framework=ModelFramework.SKLEARN,
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    },
    metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
    experiment_name="churn_prediction"
)

# Create MLflow tracking configuration
tracking_config = TrackingConfig(
    tracking_uri="databricks",
    experiment_name="churn_prediction",
    run_name=f"churn_model_{datetime.now().strftime('%Y%m%d%H%M%S')}",
    tags={
        "model_type": "classification",
        "task": "churn_prediction",
        "framework": "sklearn"
    }
)

# Load transformed data
data = pd.read_parquet("path/to/transformed_data.parquet")
X = data.drop('churned', axis=1)
y = data['churned']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
trainer = ModelTrainer(config=training_config, tracking_config=tracking_config)
result = trainer.train_and_evaluate(X_train, y_train, X_test, y_test, register_model=True)

# Check results
print(f"Training status: {result.status}")
print(f"Model URI: {result.model_uri}")
for metric_name, metric_value in result.metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")
```

## Model Deployment

Deploy models to Databricks endpoints:

```python
from databricks_mlops.models.config import DeploymentConfig
from databricks_mlops.pipelines import (ModelDeployer, DeploymentType, 
                                      ComputeType, EndpointConfig)

# Define deployment configuration with Pydantic
deployment_config = DeploymentConfig(
    model_name="customer_churn_predictor",
    model_version="1",
    endpoint_name="customer-churn-endpoint",
    deployment_type=DeploymentType.SERVING_ENDPOINT,
    compute_type=ComputeType.CPU,
    min_replicas=1,
    max_replicas=3,
    environment="staging",
    timeout_seconds=300,
    tags={
        "purpose": "churn_prediction",
        "team": "customer_analytics"
    }
)

# Deploy model
deployer = ModelDeployer(config=deployment_config)
result = deployer.deploy(model_uri="models:/customer_churn_predictor/1")

if result.status == "SUCCESS":
    print(f"Model deployed successfully to endpoint: {deployment_config.endpoint_name}")
    print(f"Endpoint URL: {result.details.get('endpoint_url')}")
else:
    print(f"Deployment failed: {result.message}")
```

## Model Monitoring

Monitor deployed models for drift and performance:

```python
from databricks_mlops.models.config import MonitoringConfig
from databricks_mlops.monitoring import DriftDetector, MetricCollector
import pandas as pd

# Define monitoring configuration with Pydantic
monitoring_config = MonitoringConfig(
    model_name="customer_churn_predictor",
    endpoint_name="customer-churn-endpoint",
    metrics=["accuracy", "drift_score", "data_quality"],
    monitor_data_drift=True,
    monitor_prediction_drift=True,
    reference_dataset_path="dbfs:/mnt/gold/feature_store/customer_churn_features/reference",
    alert_thresholds={
        "drift_score": 0.05,
        "accuracy_drop": 0.1,
        "data_quality_score": 0.8
    },
    monitoring_schedule="0 */6 * * *",  # Every 6 hours
    lookback_days=7
)

# Create drift detector and metric collector
drift_detector = DriftDetector(config=monitoring_config)
metric_collector = MetricCollector(config=monitoring_config)

# Load reference and current data
reference_data = pd.read_parquet("path/to/reference_data.parquet")
current_data = pd.read_parquet("path/to/current_data.parquet")

# Detect drift
drift_result = drift_detector.detect_drift(
    reference_data=reference_data,
    current_data=current_data
)

# Parse strongly-typed results
if drift_result.has_drift:
    print(f"Data drift detected! Overall score: {drift_result.drift_score:.4f}")
    print("Drifted features:")
    for feature in drift_result.drifted_features:
        print(f"- {feature}")
else:
    print(f"No significant data drift detected. Score: {drift_result.drift_score:.4f}")

# Collect performance metrics
y_true = current_data['churned']
y_pred = pd.read_parquet("path/to/predictions.parquet")['prediction']
timestamps = pd.Series([datetime.now()] * len(y_true))

metrics_result = metric_collector.collect_metrics(
    y_true=y_true,
    y_pred=y_pred,
    timestamps=timestamps
)

print("\nPerformance Metrics:")
for metric, value in metrics_result.metrics.items():
    print(f"- {metric}: {value:.4f}")
```

## Complete MLOps Pipeline

Orchestrate a complete end-to-end pipeline:

```python
from databricks_mlops.config import MLOpsConfigManager
from databricks_mlops.core import PipelineOrchestrator

# Create configuration manager
config_manager = MLOpsConfigManager.create_pipeline_config_manager()

# Load configurations
data_config = config_manager.load_from_yaml("data_config.yaml")
feature_config = config_manager.load_from_yaml("feature_config.yaml")
model_config = config_manager.load_from_yaml("model_config.yaml")
deployment_config = config_manager.load_from_yaml("deployment_config.yaml")
monitoring_config = config_manager.load_from_yaml("monitoring_config.yaml")

# Create orchestrator
orchestrator = PipelineOrchestrator()

# Add pipeline stages with dependencies
orchestrator.add_data_stage(
    name="data_processing",
    config=data_config,
    enabled=True
)

orchestrator.add_feature_stage(
    name="feature_engineering",
    config=feature_config,
    depends_on=["data_processing"],
    enabled=True
)

orchestrator.add_training_stage(
    name="model_training",
    config=model_config,
    depends_on=["feature_engineering"],
    enabled=True
)

orchestrator.add_deployment_stage(
    name="model_deployment",
    config=deployment_config,
    depends_on=["model_training"],
    enabled=True
)

orchestrator.add_monitoring_stage(
    name="model_monitoring",
    config=monitoring_config,
    depends_on=["model_deployment"],
    enabled=True
)

# Run the orchestrated pipeline
result = orchestrator.run()

# Check pipeline status
if result.status == "SUCCESS":
    print("Pipeline completed successfully!")
else:
    print(f"Pipeline failed at stages: {', '.join(result.failed_stages)}")
    for stage, error in result.stage_errors.items():
        print(f"{stage}: {error}")
```

## Best Practices

### Strong Type Safety

Always use Pydantic models for configuration and data structures:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional

class ModelHyperparameters(BaseModel):
    """Strong typing for model hyperparameters."""
    n_estimators: int = 100
    max_depth: Optional[int] = None
    learning_rate: float = 0.1
    
    @field_validator('n_estimators')
    def validate_n_estimators(cls, v):
        if v <= 0:
            raise ValueError("n_estimators must be positive")
        return v
    
    @field_validator('max_depth')
    def validate_max_depth(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_depth must be positive")
        return v

# Use in your code
params = ModelHyperparameters(n_estimators=200, max_depth=10, learning_rate=0.05)
```

### Error Handling

Use proper exception handling with typed exceptions:

```python
from databricks_mlops.exceptions import (ConfigurationError, ValidationError, 
                                       DeploymentError, ModelTrainingError)

try:
    # Load configuration
    config = config_manager.load_from_yaml("model_config.yaml")
    
    # Use configuration
    trainer = ModelTrainer(config=config.model_config)
    result = trainer.train_and_evaluate(X_train, y_train)
    
except ConfigurationError as e:
    print(f"Configuration error: {str(e)}")
    # Handle configuration issues
    
except ValidationError as e:
    print(f"Validation error: {str(e)}")
    # Handle validation issues
    
except ModelTrainingError as e:
    print(f"Training error: {str(e)}")
    # Handle training issues
    
except Exception as e:
    print(f"Unexpected error: {str(e)}")
    # Handle unexpected issues
```

### Configuration Management

Use environment variable substitution in configuration files:

```yaml
name: customer_churn_data_pipeline
environment: "${ENV:development}"
timeout_minutes: ${TIMEOUT:60}

data_config:
  source_path: "${DATA_SOURCE_PATH}"
  destination_path: "${DATA_DESTINATION_PATH}"
  database_config:
    warehouse_id: "${DATABRICKS_WAREHOUSE_ID}"
```

Load with environment variables:

```python
import os

# Set environment variables
os.environ["ENV"] = "production"
os.environ["TIMEOUT"] = "120"
os.environ["DATA_SOURCE_PATH"] = "dbfs:/mnt/production/raw-data/customer_data.parquet"
os.environ["DATA_DESTINATION_PATH"] = "dbfs:/mnt/production/silver/customer_data_processed"
os.environ["DATABRICKS_WAREHOUSE_ID"] = "01234567890abcdef"

# Load configuration with environment variable substitution
config = config_manager.load_from_yaml("data_config.yaml")

print(config.environment)  # Output: "production"
print(config.timeout_minutes)  # Output: 120
```

### Logging

Use structured logging throughout your application:

```python
from databricks_mlops.utils.logging import LogLevel, setup_logger

# Set up logger
logger = setup_logger("my_pipeline", LogLevel.INFO)

# Use strongly-typed log levels
logger.info("Starting pipeline execution")
logger.debug("Processing configuration parameters")

try:
    # Do something
    result = process_data(data)
    logger.info("Data processing completed", extra={"rows_processed": len(data)})
except Exception as e:
    logger.error("Error processing data", extra={"error": str(e)})
    # Handle exception
```

The framework's logging produces structured JSON logs:

```json
{
  "timestamp": "2023-05-12T15:04:23.123Z",
  "level": "INFO",
  "logger": "my_pipeline",
  "message": "Data processing completed",
  "rows_processed": 1000,
  "service": "databricks-mlops",
  "context": {
    "environment": "production",
    "pipeline": "customer_churn_data_pipeline"
  }
}
```
