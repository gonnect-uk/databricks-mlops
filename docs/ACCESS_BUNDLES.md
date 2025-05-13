# Databricks Access Bundles

This document provides detailed information about using Databricks Access Bundles for authentication and authorization in automated MLOps workflows.

## Overview

Databricks Access Bundles provide a secure, type-safe way to access Databricks resources in CI/CD pipelines and automation workflows. They enable service-to-service authentication without requiring hardcoded credentials in your code or configuration files.

## Key Features

- **Type-safe configuration**: All access bundle configurations are validated using Pydantic models
- **CI/CD integration**: Seamless integration with GitHub Actions, Azure DevOps, and other CI/CD tools
- **Environment isolation**: Separate access credentials for development, staging, and production
- **Least privilege access**: Fine-grained access control for different pipeline stages
- **Audit trail**: Comprehensive logging of all operations performed using access bundles

## Configuration Schema

Access bundles are defined using a strongly-typed YAML schema:

```yaml
name: "model-training-automation"
description: "Access bundle for automated model training pipelines"
version: "1.0.0"
target:
  workspace_url: "https://your-databricks-workspace.cloud.databricks.com"
  catalog: "ml_catalog"
  schema: "production"
auth:
  type: "service_principal"
  service_principal_id: "${DATABRICKS_SERVICE_PRINCIPAL_ID}"
  service_principal_secret: "${DATABRICKS_SERVICE_PRINCIPAL_SECRET}"
scopes:
  - name: "feature_store"
    permissions: ["READ"]
  - name: "model_registry"
    permissions: ["READ", "WRITE"]
  - name: "model_serving"
    permissions: ["READ", "WRITE"]
  - name: "jobs"
    permissions: ["READ", "WRITE", "EXECUTE"]
```

## Usage in Automation Workflows

### Model Training CI/CD

```python
from databricks_mlops.utils.auth import AccessBundleCredentials
from databricks_mlops.pipelines import ModelTrainingPipeline
from databricks_mlops.models.config import TrainingConfig

# Load credentials from access bundle
credentials = AccessBundleCredentials.from_bundle_file("/path/to/access-bundle.yaml")

# Create a type-safe pipeline with proper authentication
pipeline = ModelTrainingPipeline(
    config=TrainingConfig.from_yaml("/path/to/training-config.yaml"),
    credentials=credentials
)

# Run the pipeline with proper authentication
pipeline.run()
```

### GitHub Actions Integration

```yaml
name: Train and Deploy Model

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  train-model:
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
          uv pip install -e .[all]
          
      - name: Create access bundle
        run: |
          echo "${{ secrets.DATABRICKS_ACCESS_BUNDLE }}" > access-bundle.yaml
          
      - name: Train model
        run: |
          python -m databricks_mlops.scripts.train \
            --config configs/training-config.yaml \
            --access-bundle access-bundle.yaml
```

## Access Bundle Management

### Creating Access Bundles

Access bundles can be created using the Databricks CLI:

```bash
# Create a new access bundle
databricks access-bundles create \
  --name model-training-automation \
  --workspace-url https://your-workspace.cloud.databricks.com \
  --catalog ml_catalog \
  --schema production

# Add permissions
databricks access-bundles permissions add \
  --name model-training-automation \
  --scope model_registry \
  --permission WRITE
```

### Rotating Credentials

To rotate access bundle credentials:

```bash
# Rotate access bundle credentials
databricks access-bundles rotate-token \
  --name model-training-automation
```

## Using with the MLOps Framework

Our framework provides typed wrappers around access bundles to ensure type safety throughout the authentication process:

```python
from databricks_mlops.utils.auth import (
    AccessBundleCredentials, 
    WorkspaceAccess,
    TokenCredential
)

# Load with type validation
credentials = AccessBundleCredentials.from_bundle_file("path/to/bundle.yaml")

# Access is now available with typed methods
workspace = credentials.get_workspace_access()
catalog_access = credentials.get_catalog_access("ml_catalog")

# All operations maintain type safety
models = workspace.list_models()
```

## Security Best Practices

1. **Store securely**: Always store access bundles in a secure location (e.g., GitHub Secrets, Azure Key Vault)
2. **Rotate regularly**: Rotate credentials on a regular schedule (e.g., every 90 days)
3. **Use least privilege**: Only grant the permissions needed for each specific workflow
4. **Audit usage**: Regularly review audit logs to ensure appropriate usage
5. **Environment separation**: Use different access bundles for development, staging, and production

## Troubleshooting

### Common Issues

**Invalid Permissions**:
```
Error: User does not have permission to access resource 'models/my-model'
```

Solution: Verify the access bundle has the correct permissions for the resource.

**Expired Credentials**:
```
Error: Token has expired
```

Solution: Rotate the access bundle credentials.

## Next Steps

- See the [Automation Guide](AUTOMATION.md) for detailed CI/CD pipeline examples
- Refer to the [API Reference](../API_REFERENCE.md) for details on authentication classes
