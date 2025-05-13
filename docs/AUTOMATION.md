# MLOps Automation Guide

This document provides detailed guidance on automating MLOps workflows with the Databricks MLOps framework.

## CI/CD Integration

The framework is designed to work seamlessly with CI/CD pipelines using Databricks access bundles for secure authentication.

### GitHub Actions Pipeline Example

Here's a complete GitHub Actions workflow for automating model training, validation, and deployment:

```yaml
name: MLOps Pipeline

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy to'
        required: true
        default: 'staging'
        type: choice
        options:
          - dev
          - staging
          - production

jobs:
  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install uv
          uv pip install -e .[data,validation]
          
      - name: Create access bundle
        run: |
          echo "${{ secrets.DATABRICKS_ACCESS_BUNDLE }}" > access-bundle.yaml
          
      - name: Validate data
        run: |
          python -m databricks_mlops.scripts.validate_data \
            --config configs/data-config.yaml \
            --access-bundle access-bundle.yaml
            
      - name: Upload validation report
        uses: actions/upload-artifact@v3
        with:
          name: validation-report
          path: ./reports/validation-report.json

  feature-engineering:
    needs: data-validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install uv
          uv pip install -e .[feature-engineering]
          
      - name: Create access bundle
        run: |
          echo "${{ secrets.DATABRICKS_ACCESS_BUNDLE }}" > access-bundle.yaml
          
      - name: Generate features
        run: |
          python -m databricks_mlops.scripts.feature_engineering \
            --config configs/feature-config.yaml \
            --access-bundle access-bundle.yaml

  model-training:
    needs: feature-engineering
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install uv
          uv pip install -e .[model-training]
          
      - name: Create access bundle
        run: |
          echo "${{ secrets.DATABRICKS_ACCESS_BUNDLE }}" > access-bundle.yaml
          
      - name: Train model
        run: |
          python -m databricks_mlops.scripts.train \
            --config configs/training-config.yaml \
            --access-bundle access-bundle.yaml
            
      - name: Upload model artifacts
        uses: actions/upload-artifact@v3
        with:
          name: model-artifacts
          path: ./artifacts/models

  model-deployment:
    needs: model-training
    if: github.event_name == 'workflow_dispatch' || github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install uv
          uv pip install -e .[deployment]
          
      - name: Create access bundle
        run: |
          echo "${{ secrets.DATABRICKS_ACCESS_BUNDLE }}" > access-bundle.yaml
          
      - name: Deploy model
        run: |
          python -m databricks_mlops.scripts.deploy \
            --config configs/deployment-config.yaml \
            --access-bundle access-bundle.yaml \
            --environment ${{ github.event.inputs.environment || 'staging' }}
```

## Workflow Automation

The framework supports automating the entire MLOps lifecycle with a workflow orchestration script:

```python
from databricks_mlops.utils.auth import AccessBundleCredentials
from databricks_mlops.workflows import MLOpsWorkflow
from databricks_mlops.models.config import (
    DataConfig, 
    FeatureConfig,
    TrainingConfig,
    DeploymentConfig,
    MonitoringConfig
)

def run_mlops_workflow():
    # Load credentials from access bundle
    credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")
    
    # Create workflow with all configurations
    workflow = MLOpsWorkflow(
        credentials=credentials,
        data_config=DataConfig.from_yaml("configs/data-config.yaml"),
        feature_config=FeatureConfig.from_yaml("configs/feature-config.yaml"),
        training_config=TrainingConfig.from_yaml("configs/training-config.yaml"),
        deployment_config=DeploymentConfig.from_yaml("configs/deployment-config.yaml"),
        monitoring_config=MonitoringConfig.from_yaml("configs/monitoring-config.yaml")
    )
    
    # Run the full workflow with type safety throughout
    workflow.run()

if __name__ == "__main__":
    run_mlops_workflow()
```

## Scheduling with Databricks Jobs

You can schedule MLOps workflows using Databricks Jobs API with proper type safety:

```python
from databricks_mlops.utils.auth import AccessBundleCredentials
from databricks_mlops.utils.jobs import JobConfig, TaskConfig, ScheduleConfig

# Load credentials from access bundle
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Create job with type-safe configuration
job_config = JobConfig(
    name="daily-model-training",
    tasks=[
        TaskConfig(
            name="data-validation",
            python_script_path="databricks_mlops/scripts/validate_data.py",
            parameters={"config": "configs/data-config.yaml"}
        ),
        TaskConfig(
            name="feature-engineering",
            python_script_path="databricks_mlops/scripts/feature_engineering.py",
            parameters={"config": "configs/feature-config.yaml"},
            depends_on=["data-validation"]
        ),
        TaskConfig(
            name="model-training",
            python_script_path="databricks_mlops/scripts/train.py",
            parameters={"config": "configs/training-config.yaml"},
            depends_on=["feature-engineering"]
        ),
        TaskConfig(
            name="model-deployment",
            python_script_path="databricks_mlops/scripts/deploy.py",
            parameters={
                "config": "configs/deployment-config.yaml",
                "environment": "staging"
            },
            depends_on=["model-training"]
        )
    ],
    schedule=ScheduleConfig(
        quartz_cron_expression="0 0 0 * * ?",  # Daily at midnight
        timezone="UTC"
    )
)

# Create job with properly authenticated client
jobs_client = credentials.get_jobs_client()
job_id = jobs_client.create_job(job_config)
print(f"Created job with ID: {job_id}")
```

## Environment Variables

For environment-specific configurations, use environment variables in your YAML files:

```yaml
# deployment-config.yaml
name: "churn-prediction-model"
model_uri: "models:/churn-prediction/production"
endpoint:
  name: "churn-prediction-${ENVIRONMENT}"
  config:
    instance_type: "${INSTANCE_TYPE}"
    scale_to_zero_enabled: true
```

Then reference them in your deployment script:

```python
import os
from databricks_mlops.models.config import DeploymentConfig

# Set environment variables
os.environ["ENVIRONMENT"] = "staging"
os.environ["INSTANCE_TYPE"] = "Standard_DS3_v2"

# Load config with environment variable substitution
config = DeploymentConfig.from_yaml("configs/deployment-config.yaml")
```

## Secrets Management

The framework integrates with various secrets management solutions:

1. **GitHub Secrets**: For GitHub Actions workflows
2. **Azure Key Vault**: For Azure-based deployments
3. **AWS Secrets Manager**: For AWS-based deployments
4. **Databricks Secrets**: For Databricks Jobs

Example using Databricks Secrets:

```python
from databricks_mlops.utils.auth import AccessBundleCredentials
from databricks_mlops.utils.secrets import DBSecretsProvider

# Create a secrets provider
secrets = DBSecretsProvider(
    scope="mlops-secrets",
    credentials=AccessBundleCredentials.from_bundle_file("access-bundle.yaml")
)

# Retrieve secrets with proper typing
api_key = secrets.get_secret("model-serving-api-key")
```

## Next Steps

- See [ACCESS_BUNDLES.md](ACCESS_BUNDLES.md) for more details on authentication
- Refer to the [API Reference](../API_REFERENCE.md) for complete documentation
- Check the example CI/CD templates in the `examples/automation/` directory
