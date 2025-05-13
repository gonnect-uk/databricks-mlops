# Model Deployment Guide

This document provides comprehensive guidance on deploying models using the Databricks MLOps framework.

## Overview

The framework provides a strongly-typed approach to model deployment that ensures type safety throughout the entire deployment process. It supports deploying models to Databricks Model Registry and creating serving endpoints with proper configuration.

## Deployment Options

The framework supports multiple deployment patterns:

1. **Model Registry Deployment**: Register models in the Databricks Model Registry
2. **Serving Endpoint Deployment**: Deploy models to serving endpoints
3. **Batch Inference Deployment**: Configure models for batch inference jobs
4. **Unity Catalog Integration**: Deploy models with proper Unity Catalog permissions

## Deployment Process

### 1. Model Registration

```python
from databricks_mlops.utils.deployment import ModelRegistrar
from databricks_mlops.models.config import RegistrationConfig
from databricks_mlops.utils.auth import AccessBundleCredentials

# Load credentials for authentication
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Create type-safe registration configuration
config = RegistrationConfig(
    model_name="customer_churn_model",
    model_uri="runs:/12345/models/customer_churn",
    description="Customer churn prediction model",
    tags={
        "framework": "scikit-learn",
        "type": "classification",
        "target": "churn"
    }
)

# Register model with proper authentication and type safety
registrar = ModelRegistrar(credentials=credentials)
model_version = registrar.register_model(config)

print(f"Registered model version: {model_version}")
```

### 2. Model Promotion

```python
from databricks_mlops.utils.deployment import ModelManager
from databricks_mlops.utils.auth import AccessBundleCredentials
from databricks_mlops.models.registry import Stage

# Load credentials for authentication
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Create model manager with proper authentication
manager = ModelManager(credentials=credentials)

# Transition model to staging
manager.transition_model_version(
    name="customer_churn_model",
    version=1,
    stage=Stage.STAGING,
    archive_existing_versions=True
)

# After validation, promote to production
manager.transition_model_version(
    name="customer_churn_model",
    version=1,
    stage=Stage.PRODUCTION,
    archive_existing_versions=True
)
```

### 3. Serving Endpoint Creation

```python
from databricks_mlops.utils.deployment import ServingEndpointManager
from databricks_mlops.models.serving import EndpointConfig, TrafficConfig
from databricks_mlops.utils.auth import AccessBundleCredentials

# Load credentials for authentication
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Create endpoint manager with proper authentication
endpoint_manager = ServingEndpointManager(credentials=credentials)

# Create endpoint with type-safe configuration
endpoint_config = EndpointConfig(
    name="customer-churn-endpoint",
    model_name="customer_churn_model",
    model_version=1,
    scale_to_zero_enabled=True,
    workload_size="Small",
    workload_type="CPU",
    min_instances=1,
    max_instances=4
)

# Create endpoint with proper configuration
endpoint_id = endpoint_manager.create_endpoint(endpoint_config)

print(f"Created endpoint with ID: {endpoint_id}")
```

### 4. Traffic Management

```python
from databricks_mlops.utils.deployment import ServingEndpointManager
from databricks_mlops.models.serving import TrafficConfig
from databricks_mlops.utils.auth import AccessBundleCredentials

# Load credentials for authentication
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Create endpoint manager with proper authentication
endpoint_manager = ServingEndpointManager(credentials=credentials)

# Configure traffic split between model versions
traffic_config = TrafficConfig(
    endpoint_name="customer-churn-endpoint",
    traffic_split={
        "1": 50,  # 50% traffic to version 1
        "2": 50   # 50% traffic to version 2
    }
)

# Update traffic configuration
endpoint_manager.update_traffic(traffic_config)
```

## YAML-Based Deployment

For repeatable deployments, use YAML configuration:

```yaml
# deployment-config.yaml
name: "customer-churn-deployment"
model:
  name: "customer_churn_model"
  uri: "runs:/12345/models/customer_churn"
  description: "Customer churn prediction model"
  tags:
    framework: "scikit-learn"
    type: "classification"
    target: "churn"

serving_endpoint:
  name: "customer-churn-endpoint"
  workload_size: "Small"
  workload_type: "CPU"
  scale_to_zero_enabled: true
  min_instances: 1
  max_instances: 4
  
stages:
  - name: "staging"
    auto_promote: true
    promotion_criteria:
      metrics:
        - name: "accuracy"
          threshold: 0.85
        - name: "f1_score"
          threshold: 0.80
      tests:
        - "data_drift_check"
        - "model_signature_check"
  
  - name: "production"
    auto_promote: false
    traffic_config:
      blue_green: true
      initial_traffic_percentage: 10
      increments: [10, 30, 50, 100]
      evaluation_period_minutes: 60
```

To deploy using this configuration:

```python
from databricks_mlops.utils.deployment import DeploymentManager
from databricks_mlops.models.config import DeploymentConfig
from databricks_mlops.utils.auth import AccessBundleCredentials

# Load credentials for authentication
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Load type-safe configuration
config = DeploymentConfig.from_yaml("deployment-config.yaml")

# Create deployment manager with proper authentication
manager = DeploymentManager(credentials=credentials)

# Deploy the model with type safety throughout
deployment_id = manager.deploy(config)

print(f"Deployment ID: {deployment_id}")
```

## Continuous Deployment

The framework supports continuous deployment patterns:

### Blue-Green Deployment

```python
from databricks_mlops.utils.deployment import BlueGreenDeployment
from databricks_mlops.models.config import BlueGreenConfig
from databricks_mlops.utils.auth import AccessBundleCredentials

# Load credentials for authentication
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Create blue-green deployment configuration
config = BlueGreenConfig(
    endpoint_name="customer-churn-endpoint",
    new_model_version=2,
    current_model_version=1,
    initial_traffic_percentage=10,
    evaluation_period_minutes=60,
    traffic_increments=[10, 30, 50, 100],
    rollback_threshold_metrics={
        "error_rate": 0.05  # Rollback if error rate exceeds 5%
    }
)

# Create blue-green deployment manager
deployment = BlueGreenDeployment(
    config=config,
    credentials=credentials
)

# Start the blue-green deployment
deployment.start()

# Monitor the deployment (usually done in a separate process or job)
status = deployment.check_status()
print(f"Deployment status: {status}")
```

### Canary Deployment

```python
from databricks_mlops.utils.deployment import CanaryDeployment
from databricks_mlops.models.config import CanaryConfig
from databricks_mlops.utils.auth import AccessBundleCredentials

# Load credentials for authentication
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Create canary deployment configuration
config = CanaryConfig(
    endpoint_name="customer-churn-endpoint",
    new_model_version=2,
    current_model_version=1,
    canary_traffic_percentage=5,
    evaluation_period_minutes=120,
    success_criteria={
        "accuracy": {
            "min_threshold": 0.85,
            "comparison": ">=",
            "baseline_version": 1
        },
        "latency_ms": {
            "max_threshold": 100,
            "comparison": "<=",
            "baseline_version": 1
        }
    }
)

# Create canary deployment manager
deployment = CanaryDeployment(
    config=config,
    credentials=credentials
)

# Start the canary deployment
deployment.start()

# Monitor the deployment (usually done in a separate process or job)
status = deployment.check_status()
print(f"Deployment status: {status}")
```

## Deployment Hooks

The framework supports pre and post-deployment hooks for validation, notification, and integration:

```python
from databricks_mlops.utils.deployment import DeploymentManager
from databricks_mlops.models.config import DeploymentConfig
from databricks_mlops.hooks import SlackNotifier, ModelValidator
from databricks_mlops.utils.auth import AccessBundleCredentials

# Load credentials for authentication
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Load type-safe configuration
config = DeploymentConfig.from_yaml("deployment-config.yaml")

# Create hooks
validators = [
    ModelValidator(metric_thresholds={"accuracy": 0.85, "f1": 0.80})
]

notifiers = [
    SlackNotifier(webhook_url="https://hooks.slack.com/services/XXX/YYY/ZZZ")
]

# Create deployment manager with hooks
manager = DeploymentManager(
    credentials=credentials,
    pre_deployment_hooks=validators,
    post_deployment_hooks=notifiers
)

# Deploy the model with hooks
deployment_id = manager.deploy(config)
```

## Command-Line Deployment

The framework provides command-line utilities for deployment:

```bash
# Deploy a model from the command line
python -m databricks_mlops.scripts.deploy \
  --config deployment-config.yaml \
  --access-bundle access-bundle.yaml \
  --environment production \
  --notify slack
```

## Multi-Environment Deployment

For deploying across environments:

```yaml
# multi-env-deployment.yaml
name: "customer-churn-deployment"
model:
  name: "customer_churn_model"
  uri: "runs:/12345/models/customer_churn"

environments:
  dev:
    serving_endpoint:
      name: "customer-churn-dev"
      workload_size: "Small"
      scale_to_zero_enabled: true
      
  staging:
    serving_endpoint:
      name: "customer-churn-staging"
      workload_size: "Medium"
      scale_to_zero_enabled: true
      min_instances: 1
      
  production:
    serving_endpoint:
      name: "customer-churn-prod"
      workload_size: "Large"
      scale_to_zero_enabled: false
      min_instances: 2
      max_instances: 8
```

To deploy using this configuration:

```python
from databricks_mlops.utils.deployment import MultiEnvironmentDeployment
from databricks_mlops.models.config import MultiEnvConfig
from databricks_mlops.utils.auth import AccessBundleCredentials

# Load credentials for authentication
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Load type-safe configuration
config = MultiEnvConfig.from_yaml("multi-env-deployment.yaml")

# Create deployment manager
deployment = MultiEnvironmentDeployment(
    config=config,
    credentials=credentials
)

# Deploy to specific environment
deployment.deploy(environment="staging")
```

## Troubleshooting Deployments

The framework provides utilities for diagnosing deployment issues:

```python
from databricks_mlops.utils.deployment import DeploymentDiagnostics
from databricks_mlops.utils.auth import AccessBundleCredentials

# Load credentials for authentication
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Create diagnostics tool
diagnostics = DeploymentDiagnostics(credentials=credentials)

# Check deployment health
health_report = diagnostics.check_endpoint_health("customer-churn-endpoint")
print(health_report)

# Get detailed logs
logs = diagnostics.get_endpoint_logs("customer-churn-endpoint", lines=100)
print(logs)
```

## Best Practices

1. **Environment Isolation**: Use separate endpoints for development, staging, and production
2. **Version Control**: Keep deployment configurations in version control
3. **Credential Management**: Use access bundles for secure authentication
4. **Gradual Rollout**: Use blue-green or canary deployments for critical models
5. **Monitoring**: Always set up monitoring before deploying to production
6. **Rollback Plan**: Have clear procedures for rolling back problematic deployments

## Next Steps

- See [MODEL_SERVING.md](MODEL_SERVING.md) for details on accessing deployed models
- Refer to [MONITORING.md](MONITORING.md) for post-deployment monitoring guidance
- Check [ACCESS_BUNDLES.md](ACCESS_BUNDLES.md) for authentication best practices
