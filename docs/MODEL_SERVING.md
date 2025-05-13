# Model Serving Guide

This document provides comprehensive guidance on accessing and utilizing deployed models through Databricks model serving endpoints using the framework.

## Overview

The framework provides strongly-typed clients for interacting with Databricks model serving endpoints, ensuring type safety throughout the request-response cycle. It focuses on providing reliable, type-safe access to deployed models for both real-time inference and batch prediction scenarios.

## Model Serving Concepts

### Endpoint Types

The framework supports various types of model serving endpoints:

1. **Real-time Serving**: Low-latency API endpoints for online inference
2. **Serverless Endpoints**: Scale-to-zero endpoints for cost-efficient serving
3. **Custom Containers**: Support for custom Docker containers
4. **Batch Inference**: Scheduled or on-demand batch prediction jobs

### Type-Safe Client Architecture

The client architecture ensures strong typing throughout:

```
┌─────────────┐     ┌───────────────┐     ┌─────────────────┐
│ Pydantic    │     │ Type-Safe     │     │ Databricks      │
│ Request     │────▶│ Serialization │────▶│ Model Serving   │
│ Models      │     │ Layer         │     │ REST API        │
└─────────────┘     └───────────────┘     └─────────────────┘
       ▲                                          │
       │                                          │
       │           ┌───────────────┐              │
       │           │ Type-Safe     │              │
       └───────────│ Deserialization│◀─────────────┘
                   │ Layer         │
                   └───────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Pydantic    │
                   │ Response    │
                   │ Models      │
                   └─────────────┘
```

## Using the Model Serving Clients

### Tabular Model Client

For classification and regression models:

```python
from databricks_mlops.utils.model_serving import (
    TabularModelClient, EndpointCredentials, AuthType
)
import pandas as pd

# Create strongly-typed credentials
credentials = EndpointCredentials(
    auth_type=AuthType.TOKEN,
    token="your-databricks-token"
)

# Initialize the client with proper typing
client = TabularModelClient(
    workspace_url="https://your-workspace.cloud.databricks.com",
    credentials=credentials
)

# Prepare feature data with correct types
feature_data = pd.DataFrame({
    "tenure": [12, 24, 36],
    "monthly_charges": [50.0, 70.0, 90.0],
    "contract_type": ["Month-to-month", "One year", "Two year"],
    "total_charges": [600.0, 1680.0, 3240.0]
})

# Get predictions with full type safety
predictions = client.predict(
    endpoint_name="customer-churn-classifier",
    features=feature_data
)

# Process results with proper DataFrame typing
probabilities = predictions['probability']
predictions = predictions['prediction']
```

### Batch Inference

For large-scale batch prediction scenarios:

```python
from databricks_mlops.utils.model_serving import BatchInferenceJob
from databricks_mlops.models.serving import BatchConfig
from databricks_mlops.utils.auth import AccessBundleCredentials

# Load credentials for authentication
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Create batch configuration
batch_config = BatchConfig(
    model_name="customer_churn_model",
    model_version=1,
    input_path="dbfs:/data/customer_features",
    output_path="dbfs:/data/customer_predictions",
    compute_config={
        "num_workers": 4,
        "instance_type": "Standard_DS3_v2"
    }
)

# Create and run batch job
batch_job = BatchInferenceJob(
    config=batch_config,
    credentials=credentials
)

# Submit the job
job_id = batch_job.submit()
print(f"Submitted batch job with ID: {job_id}")

# Wait for completion
batch_job.wait_for_completion()

# Get results path
results_path = batch_job.get_results_path()
print(f"Results available at: {results_path}")
```

## Performance Optimization

### Connection Pooling

To optimize connection management for high-throughput scenarios:

```python
from databricks_mlops.utils.model_serving import (
    ConnectionPoolConfig, 
    TabularModelClient
)

# Create connection pool configuration
pool_config = ConnectionPoolConfig(
    max_connections=20,
    max_connections_per_endpoint=5,
    connection_timeout_ms=5000,
    keep_alive_ms=30000
)

# Create client with connection pooling
client = TabularModelClient(
    workspace_url="https://your-workspace.cloud.databricks.com",
    credentials=credentials,
    connection_pool=pool_config
)
```

### Batching Requests

For efficient batching of predictions:

```python
import pandas as pd
from databricks_mlops.utils.model_serving import TabularModelClient, BatchConfig

# Prepare large feature dataset
features = pd.read_parquet("large_feature_set.parquet")

# Configure batching
batch_config = BatchConfig(
    batch_size=1000,  # Records per batch
    max_workers=4,    # Parallel workers
    timeout_ms=30000  # Timeout per batch
)

# Create client with batching configuration
client = TabularModelClient(
    workspace_url="https://your-workspace.cloud.databricks.com",
    credentials=credentials,
    batch_config=batch_config
)

# Process predictions in batches
predictions = client.predict_batch(
    endpoint_name="customer-churn-classifier",
    features=features
)
```

## Monitoring and Observability

Track endpoint performance and request metrics:

```python
from databricks_mlops.utils.model_serving import ModelServingMonitor
from databricks_mlops.utils.auth import AccessBundleCredentials

# Load credentials for authentication
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Create monitoring client
monitor = ModelServingMonitor(credentials=credentials)

# Get endpoint metrics
metrics = monitor.get_endpoint_metrics(
    endpoint_name="customer-churn-endpoint",
    start_time="2025-05-01T00:00:00Z",
    end_time="2025-05-13T00:00:00Z",
    metrics=["requests", "latency_p95", "error_rate"]
)

# Get detailed request logs
logs = monitor.get_request_logs(
    endpoint_name="customer-churn-endpoint",
    limit=100,
    filter_expression="status_code >= 400"
)
```

## Error Handling

The framework provides robust, type-safe error handling:

```python
from databricks_mlops.utils.model_serving import TabularModelClient
from databricks_mlops.exceptions import (
    EndpointNotFoundError,
    ModelServingTimeoutError,
    ModelServingAuthError,
    InvalidPredictionInput
)
import pandas as pd

# Create client
client = TabularModelClient(
    workspace_url="https://your-workspace.cloud.databricks.com",
    credentials=credentials
)

# Feature data
features = pd.DataFrame({
    "tenure": [12, 24, 36],
    "monthly_charges": [50.0, 70.0, 90.0]
})

try:
    predictions = client.predict(
        endpoint_name="customer-churn-classifier",
        features=features
    )
except EndpointNotFoundError as e:
    print(f"Endpoint not found: {e}")
except ModelServingTimeoutError as e:
    print(f"Request timed out: {e}")
except ModelServingAuthError as e:
    print(f"Authentication error: {e}")
except InvalidPredictionInput as e:
    print(f"Invalid input data: {e}")
```

## Testing and Mocking

For unit testing purposes, the framework provides mocks:

```python
from databricks_mlops.testing.mocks import MockTabularModelClient
import pandas as pd
import pytest

@pytest.fixture
def mock_client():
    # Create mock client with predefined responses
    return MockTabularModelClient(
        mock_responses={
            "customer-churn-endpoint": {
                "prediction": [0, 1, 0],
                "probability": [0.2, 0.8, 0.3]
            }
        }
    )

def test_prediction_workflow(mock_client):
    # Test features
    features = pd.DataFrame({
        "tenure": [12, 24, 36],
        "monthly_charges": [50.0, 70.0, 90.0]
    })
    
    # Get predictions from mock
    predictions = mock_client.predict(
        endpoint_name="customer-churn-endpoint",
        features=features
    )
    
    # Verify predictions
    assert 'prediction' in predictions
    assert 'probability' in predictions
    assert len(predictions['prediction']) == 3
```

## Command-Line Interface

The framework provides a command-line interface for model serving interactions:

```bash
# Get predictions from an endpoint
python -m databricks_mlops.cli.predict \
  --endpoint customer-churn-endpoint \
  --input features.parquet \
  --output predictions.parquet \
  --access-bundle access-bundle.yaml

# Get endpoint metrics
python -m databricks_mlops.cli.serving_metrics \
  --endpoint customer-churn-endpoint \
  --metric latency_p95 \
  --days 7 \
  --access-bundle access-bundle.yaml
```

## Configuration Management

Manage endpoint configurations with YAML:

```yaml
# serving-config.yaml
endpoints:
  - name: customer-churn-endpoint
    model_name: customer_churn_model
    model_version: 1
    scale_to_zero_enabled: true
    min_instances: 1
    max_instances: 4
    
  - name: customer-segmentation-endpoint
    model_name: customer_segmentation_model
    model_version: 2
    scale_to_zero_enabled: false
    min_instances: 2
    max_instances: 8

client_config:
  connection_timeout_ms: 5000
  retry_count: 3
  batch_size: 1000
```

Load this configuration:

```python
from databricks_mlops.utils.model_serving import ServingConfig, ModelServingManager
from databricks_mlops.utils.auth import AccessBundleCredentials

# Load credentials
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Load serving configuration
config = ServingConfig.from_yaml("serving-config.yaml")

# Create serving manager
manager = ModelServingManager(
    config=config,
    credentials=credentials
)

# Get client for specific endpoint
client = manager.get_client("customer-churn-endpoint")

# Use client for predictions
predictions = client.predict(features=features)
```

## Integration with Model Registry

Integrate model serving with model registry for versioning:

```python
from databricks_mlops.utils.model_serving import RegistryIntegratedClient
from databricks_mlops.utils.auth import AccessBundleCredentials

# Load credentials
credentials = AccessBundleCredentials.from_bundle_file("access-bundle.yaml")

# Create registry-aware client
client = RegistryIntegratedClient(
    model_name="customer_churn_model",
    stage="production",  # Use production version
    credentials=credentials
)

# Client automatically uses latest production model
predictions = client.predict(features=features)

# Check which model version was used
version = client.get_current_model_version()
print(f"Used model version: {version}")
```

## Next Steps

- See [DEPLOYMENT.md](DEPLOYMENT.md) for model deployment details
- Refer to [MONITORING.md](MONITORING.md) for post-deployment monitoring
- Check [ACCESS_BUNDLES.md](ACCESS_BUNDLES.md) for authentication best practices
