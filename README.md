<div align="center">

```
       ████████╗██╗   ██╗██████╗ ███████╗███████╗ █████╗ ███████╗███████╗       
      ╚══██╔══╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔════╝       
         ██║    ╚████╔╝ ██████╔╝█████╗  ███████╗███████║█████╗  █████╗         
         ██║     ╚██╔╝  ██╔═══╝ ██╔══╝  ╚════██║██╔══██║██╔══╝  ██╔══╝         
         ██║      ██║   ██║     ███████╗███████║██║  ██║██║     ███████╗       
         ╚═╝      ╚═╝   ╚═╝     ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝       
                                                                               
██████╗  █████╗ ████████╗ █████╗ ██████╗ ██████╗ ██╗ ██████╗██╗  ██╗███████╗   
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║██╔════╝██║ ██╔╝██╔════╝   
██║  ██║███████║   ██║   ███████║██████╔╝██████╔╝██║██║     █████╔╝ ███████╗   
██║  ██║██╔══██║   ██║   ██╔══██║██╔══██╗██╔══██╗██║██║     ██╔═██╗ ╚════██║   
██████╔╝██║  ██║   ██║   ██║  ██║██████╔╝██║  ██║██║╚██████╗██║  ██╗███████║   
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝   
                                                                               
    ███╗   ███╗██╗      ██████╗ ██████╗ ███████╗                               
    ████╗ ████║██║     ██╔═══██╗██╔══██╗██╔════╝                               
    ██╔████╔██║██║     ██║   ██║██████╔╝███████╗                               
    ██║╚██╔╝██║██║     ██║   ██║██╔═══╝ ╚════██║                               
    ██║ ╚═╝ ██║███████╗╚██████╔╝██║     ███████║                               
    ╚═╝     ╚═╝╚══════╝ ╚═════╝ ╚═╝     ╚══════╝                               
```

# Databricks MLOps Framework

### A strongly-typed, Pydantic-based MLOps framework for production ML on Databricks

[![License](https://img.shields.io/github/license/gonnect-uk/databricks-mlops)](https://github.com/gonnect-uk/databricks-mlops/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/gonnect-uk/databricks-mlops)](https://github.com/gonnect-uk/databricks-mlops/stargazers)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://pypi.org/project/databricks-mlops/)
[![Type Checking](https://img.shields.io/badge/type%20checking-mypy-brightgreen)]()
[![Code Style](https://img.shields.io/badge/code%20style-black-black)]()

</div>

## Overview

A comprehensive, type-safe MLOps framework for Databricks that follows best practices for the end-to-end machine learning lifecycle. The framework emphasizes type safety, modularity, and automated MLOps processes.

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [**API Reference**](API_REFERENCE.md) | Comprehensive API reference for all framework components |
| [**Usage Guide**](USAGE_GUIDE.md) | Step-by-step guide to using the framework with examples |
| [**Architecture Overview**](ARCHITECTURE.md) | Detailed architecture and design principles |

## 🔍 Key Features

- **Strong Type Safety**: Built from the ground up with Pydantic models and Python type annotations
- **Modular Pipeline Components**: Standardized interfaces for data, feature, training, deployment, and monitoring pipelines
- **Configuration-Driven**: All behavior controlled through strongly-typed YAML configurations
- **Databricks Integration**: Seamless integration with Databricks Delta tables, MLflow, and model serving
- **Monitoring & Drift Detection**: Built-in components for model monitoring and drift detection
- **Validation**: Data validation at every step of the ML pipeline

## 💾 Examples

The `examples/` directory contains complete implementations showcasing the framework:

| Example | Description |
|---------|-------------|
| [**Complete MLOps Pipeline**](examples/complete_mlops_pipeline.py) | End-to-end MLOps pipeline orchestration |
| [**Customer Churn Prediction**](examples/customer_churn_prediction.py) | Practical implementation of churn prediction model |
| [**Model Monitoring**](examples/model_monitoring_example.py) | Drift detection and monitoring implementation |
| [**Integration Tests**](examples/integration_test.py) | Tests demonstrating component interactions |

## 📲 Installation

We recommend using [uv](https://github.com/astral-sh/uv) for faster, more reliable Python package management:

```bash
# From PyPI
uv pip install databricks-mlops

# From GitHub
uv pip install git+https://github.com/gonnect-uk/databricks-mlops.git

# Development installation
git clone https://github.com/gonnect-uk/databricks-mlops.git
cd databricks-mlops
uv pip install -e .

# Install with all dependencies
uv pip install "databricks-mlops[all]"
```

You can also use traditional pip if uv is not available:

```bash
pip install databricks-mlops
```

## Overview

This framework provides a standardized approach to machine learning operations on Databricks, incorporating:

- Strong typing with Pydantic models
- End-to-end ML pipeline automation
- Built-in monitoring and data quality checks
- Comprehensive CI/CD integration
- Multi-environment deployment orchestration
- Artifact management and versioning

## 🔐 Type Safety Philosophy

This framework is built on a foundation of aggressive type safety and pure Pydantic models:

- **No Direct JSON Handling**: All data structures are proper Pydantic models
- **No String Manipulation**: Proper type casting and validation instead of string parsing
- **No `hasattr()` Checks**: Explicit model attributes with proper defaults
- **Domain-Driven Design**: Components map directly to MLOps domain concepts
- **Specialized Exceptions**: Clear error types instead of generic exceptions
- **Functional Approach**: Helper methods and immutable data where possible
- **Robust Fallbacks**: Proper default values and clear error handling

Every component in the framework follows this philosophy, ensuring maximum type safety with minimal runtime errors.

## Architecture

```mermaid
flowchart TB
    subgraph "Databricks MLOps Framework"
    direction TB
        subgraph DataIngestion["Data Ingestion"]
            DI_Raw["Raw Data Sources"]
            DI_Validation["Data Validation"]
            DI_Transformation["Data Transformation"]
            DI_Raw --> DI_Validation --> DI_Transformation
        end

        subgraph FeatureStore["Feature Engineering"]
            FE_Processing["Feature Processing"]
            FE_Storage["Feature Storage"]
            FE_Registry["Feature Registry"]
            FE_Processing --> FE_Storage --> FE_Registry
        end

        subgraph ModelDevelopment["Model Development"]
            MD_Exp["Experimentation"]
            MD_Training["Training"]
            MD_Evaluation["Evaluation"]
            MD_Registry["Model Registry"]
            MD_Exp --> MD_Training --> MD_Evaluation --> MD_Registry
        end

        subgraph ModelDeployment["Model Deployment"]
            MDEP_Promotion["Model Promotion"]
            MDEP_Staging["Staging Deployment"]
            MDEP_Production["Production Deployment"]
            MDEP_Promotion --> MDEP_Staging --> MDEP_Production
        end

        subgraph ModelMonitoring["Model Monitoring"]
            MM_Metrics["Performance Metrics"]
            MM_Drift["Drift Detection"]
            MM_Alerts["Alerting"]
            MM_Metrics --> MM_Drift --> MM_Alerts
        end

        DataIngestion --> FeatureStore
        FeatureStore --> ModelDevelopment
        ModelDevelopment --> ModelDeployment
        ModelDeployment --> ModelMonitoring
        ModelMonitoring -.-> DataIngestion
    end

    subgraph "External Systems"
        Git["Git Repository"]
        CI["CI/CD System"]
        DS["Data Sources"]
        Consumers["Consumers/Applications"]
    end

    DS --> DataIngestion
    Git <--> DataIngestion
    Git <--> FeatureStore
    Git <--> ModelDevelopment
    Git <--> ModelDeployment
    Git <--> ModelMonitoring
    CI <--> ModelDeployment
    ModelDeployment --> Consumers
```

## Component Structure

```mermaid
classDiagram
    class Pipeline {
        +PipelineConfig config
        +run() Task
        +validate() ValidationResult
    }
    
    class DataPipeline {
        +DataConfig config
        +ingest() DataFrame
        +validate() DataQualityResult
        +transform() DataFrame
        +save() Result
    }
    
    class FeaturePipeline {
        +FeatureConfig config
        +extract_features() Features
        +register_features() RegistryResult
    }
    
    class TrainingPipeline {
        +TrainingConfig config
        +prepare_training_data() TrainingData
        +train() TrainingResult
        +evaluate() EvaluationMetrics
        +register_model() RegistryResult
    }
    
    class DeploymentPipeline {
        +DeploymentConfig config
        +promote_model() PromotionResult
        +deploy() DeploymentResult
        +verify() VerificationResult
    }
    
    class MonitoringPipeline {
        +MonitoringConfig config
        +collect_metrics() MetricsResult
        +detect_drift() DriftResult
        +trigger_alerts() AlertResult
    }
    
    Pipeline <|-- DataPipeline
    Pipeline <|-- FeaturePipeline
    Pipeline <|-- TrainingPipeline
    Pipeline <|-- DeploymentPipeline
    Pipeline <|-- MonitoringPipeline
```

## Data Flow

```mermaid
flowchart LR
    subgraph "Data Lake"
        Bronze["Bronze Layer\n(Raw Data)"]
        Silver["Silver Layer\n(Cleaned Data)"]
        Gold["Gold Layer\n(Feature Tables)"]
        Bronze --> Silver --> Gold
    end
    
    subgraph "ML Platform"
        FS["Feature Store"]
        TR["Training Jobs"]
        MR["Model Registry"]
        Gold --> FS --> TR --> MR
    end
    
    subgraph "Deployment"
        ST["Staging"]
        PR["Production"]
        MR --> ST --> PR
    end
    
    subgraph "Monitoring"
        MD["Metric Dashboard"]
        DD["Drift Detection"]
        AL["Alerts"]
        PR --> MD --> DD --> AL
        AL -.-> Bronze
    end
```

## Workflow Orchestration

```mermaid
stateDiagram-v2
    [*] --> DataIngestion
    DataIngestion --> DataValidation
    
    state DataValidation {
        [*] --> RunDataTests
        RunDataTests --> DecideOnData
        DecideOnData --> [*]: Valid
        DecideOnData --> FixDataIssues: Invalid
        FixDataIssues --> RunDataTests
    }
    
    DataValidation --> FeatureEngineering
    FeatureEngineering --> ModelTraining
    
    state ModelTraining {
        [*] --> Experiment
        Experiment --> EvaluateModel
        EvaluateModel --> DecideOnModel
        DecideOnModel --> RegisterModel: Meets criteria
        DecideOnModel --> Experiment: Doesn't meet criteria
        RegisterModel --> [*]
    }
    
    ModelTraining --> ModelDeployment
    
    state ModelDeployment {
        [*] --> DeployToStaging
        DeployToStaging --> RunStagingTests
        RunStagingTests --> PromoteToProduction: Passes tests
        RunStagingTests --> FixDeploymentIssues: Fails tests
        FixDeploymentIssues --> DeployToStaging
        PromoteToProduction --> [*]
    }
    
    ModelDeployment --> ModelMonitoring
    
    state ModelMonitoring {
        [*] --> CollectMetrics
        CollectMetrics --> AnalyzePerformance
        AnalyzePerformance --> CheckDrift
        CheckDrift --> TriggerAlert: Drift detected
        CheckDrift --> CollectMetrics: No issues
        TriggerAlert --> StartRetraining
        StartRetraining --> [*]
    }
    
    ModelMonitoring --> [*]
```

## Getting Started

### Installation

```bash
# Using uv (recommended)
uv venv -p python3.9 .venv
source .venv/bin/activate
uv pip install -e .

# Or using pip
pip install -e .
```

### Basic Usage

Here's a simple example to get started:

```python
from databricks_mlops.core.pipeline import DataPipeline
from databricks_mlops.models.config import DataConfig
from databricks_mlops.utils.logging import setup_logger

# Setup logger
logger = setup_logger("data-pipeline")

# Create configuration
config = DataConfig(
    source_path="dbfs:/path/to/source",
    destination_path="dbfs:/path/to/destination",
    table_name="my_dataset",
    validation_rules=[
        "column_count > 5",
        "customers.id is not null"
    ]
)

# Initialize and run pipeline
pipeline = DataPipeline(config)
result = pipeline.run()

logger.info(f"Pipeline completed with status: {result.status}")
```

## Development

### Running Tests

```bash
pytest
```

### Code Quality Checks

```bash
# Run all quality checks
pre-commit run --all-files

# Or individual checks
black databricks_mlops tests
isort databricks_mlops tests
mypy databricks_mlops
ruff check databricks_mlops tests
```

## License

MIT
