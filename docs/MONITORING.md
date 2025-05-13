# Model Monitoring

This document provides comprehensive guidance on monitoring models deployed with the Databricks MLOps framework.

## Overview

Model monitoring is critical for ensuring that machine learning models continue to perform as expected in production. The framework provides robust monitoring capabilities with a focus on type safety and integration with Databricks.

## Monitoring Components

The monitoring system consists of several key components:

1. **Data Drift Detection**: Monitor changes in feature distributions
2. **Model Performance Tracking**: Track metrics like accuracy, F1-score, etc.
3. **Prediction Drift Detection**: Monitor changes in model outputs
4. **Alerting System**: Configure alerts for performance degradation
5. **Dashboards**: Visualize monitoring metrics

## Monitoring Architecture

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Reference     │     │ Current       │     │ Metrics       │
│ Data          │────▶│ Data          │────▶│ Storage       │
│ (Baseline)    │     │ (Production)  │     │ (Delta)       │
└───────────────┘     └───────────────┘     └───────────────┘
       │                     │                      │
       │                     │                      │
       ▼                     ▼                      ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Drift         │     │ Performance   │     │ Alert         │
│ Detection     │────▶│ Evaluation    │────▶│ System        │
│ Engine        │     │ Engine        │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
                                                    │
                                                    │
                                                    ▼
                                            ┌───────────────┐
                                            │ Notification  │
                                            │ System        │
                                            │ (Email/Slack) │
                                            └───────────────┘
```

## Setting Up Monitoring

### Basic Configuration

```python
from databricks_mlops.monitoring import ModelMonitor
from databricks_mlops.models.monitoring import MonitoringConfig
from databricks_mlops.utils.auth import WorkspaceConfig

# Create workspace configuration
workspace = WorkspaceConfig(
    host="https://your-workspace.cloud.databricks.com",
    token="${DATABRICKS_TOKEN}"
)

# Create monitoring configuration
config = MonitoringConfig(
    model_name="customer_churn_predictor",
    reference_dataset="dbfs:/reference/churn_baseline.delta",
    metrics=["accuracy", "f1_score", "precision", "recall"]
)

# Create monitor
monitor = ModelMonitor(workspace=workspace)

# Set up monitoring
job_id = monitor.setup_monitoring(config=config)
print(f"Monitoring job created with ID: {job_id}")
```

### Advanced Configuration

For more complex scenarios, the framework supports detailed configuration options:

```python
from databricks_mlops.monitoring import ModelMonitor
from databricks_mlops.models.monitoring import (
    MonitoringConfig, 
    DriftConfig, 
    AlertConfig,
    MetricThreshold,
    DriftMethod
)

# Create comprehensive monitoring configuration
config = MonitoringConfig(
    model_name="customer_churn_predictor",
    reference_dataset="dbfs:/reference/churn_baseline.delta",
    current_dataset="dbfs:/production/churn_current.delta",
    metrics=["accuracy", "f1_score", "precision", "recall", "roc_auc"],
    
    # Configure feature drift detection
    feature_drift=DriftConfig(
        features=["tenure", "monthly_charges", "total_charges", "contract_type"],
        drift_methods=[
            DriftMethod.WASSERSTEIN,
            DriftMethod.KS_TEST,
            DriftMethod.JS_DIVERGENCE
        ],
        threshold=0.05,
        categorical_features=["contract_type", "payment_method"]
    ),
    
    # Configure prediction drift detection
    prediction_drift=DriftConfig(
        drift_methods=[DriftMethod.PSI],
        threshold=0.1
    ),
    
    # Configure alerting
    alerts=AlertConfig(
        email_recipients=["data-science@example.com", "ml-ops@example.com"],
        slack_webhook="https://hooks.slack.com/services/XXX/YYY/ZZZ",
        thresholds={
            "accuracy": MetricThreshold(min_value=0.85, comparison=">="),
            "f1_score": MetricThreshold(min_value=0.80, comparison=">="),
            "drift_score": MetricThreshold(max_value=0.10, comparison="<=")
        }
    ),
    
    # Configure schedule
    schedule="0 */6 * * *",  # Every 6 hours
    
    # Additional settings
    compute_config={
        "instance_type": "Standard_DS3_v2",
        "num_workers": 2
    },
    
    dashboard_name="churn_model_monitoring"
)

# Create monitor
monitor = ModelMonitor(workspace=workspace)

# Set up monitoring with full type safety
job_id = monitor.setup_monitoring(config=config)
```

## Drift Detection Methods

The framework supports multiple statistical methods for detecting drift:

| Method | Description | Best For |
|--------|-------------|----------|
| **Wasserstein Distance** | Measures the "earth mover's distance" between distributions | Numerical features |
| **Kolmogorov-Smirnov (KS) Test** | Statistical test for equality of distributions | Numerical features |
| **Jensen-Shannon (JS) Divergence** | Measure of similarity between distributions | Probability distributions |
| **Population Stability Index (PSI)** | Measures shift in distributions | Categorical features |
| **Chi-Square Test** | Statistical test for independence | Categorical features |

## Monitoring Metrics

### Data Quality Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Missing Values Rate** | Percentage of missing values | `sum(is_null) / count(*)` |
| **Outlier Rate** | Percentage of outliers | `sum(is_outlier) / count(*)` |
| **Invalid Values Rate** | Percentage of invalid values | `sum(is_invalid) / count(*)` |

### Statistical Metrics

| Metric | Description |
|--------|-------------|
| **Mean** | Average value of a feature |
| **Median** | Middle value of a feature |
| **Standard Deviation** | Measure of dispersion |
| **Min/Max** | Minimum and maximum values |
| **Correlation** | Correlation between features |

### Model Performance Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Accuracy** | Overall correctness | Classification |
| **Precision** | Positive prediction accuracy | Classification |
| **Recall** | Sensitivity/True Positive Rate | Classification |
| **F1-Score** | Harmonic mean of precision and recall | Classification |
| **ROC AUC** | Area under ROC curve | Classification |
| **MAE** | Mean Absolute Error | Regression |
| **MSE** | Mean Squared Error | Regression |
| **RMSE** | Root Mean Squared Error | Regression |
| **R²** | Coefficient of determination | Regression |

## Alerting System

The framework provides a flexible alerting system:

```python
from databricks_mlops.monitoring import AlertManager
from databricks_mlops.models.monitoring import Alert, Severity, Channel

# Create alert manager
alert_manager = AlertManager(workspace=workspace)

# Define custom alerts
alerts = [
    Alert(
        name="accuracy_drop",
        description="Model accuracy dropped below threshold",
        condition="accuracy < 0.85",
        severity=Severity.HIGH,
        channels=[
            Channel(type="email", recipients=["data-science@example.com"]),
            Channel(type="slack", webhook="https://hooks.slack.com/services/XXX/YYY/ZZZ")
        ]
    ),
    Alert(
        name="drift_detected",
        description="Significant feature drift detected",
        condition="drift_score > 0.1",
        severity=Severity.MEDIUM,
        channels=[
            Channel(type="email", recipients=["ml-ops@example.com"])
        ]
    )
]

# Register alerts
for alert in alerts:
    alert_manager.register_alert(alert)
```

## Monitoring Dashboard

The framework can generate comprehensive monitoring dashboards:

```python
from databricks_mlops.monitoring import DashboardGenerator
from databricks_mlops.models.monitoring import DashboardConfig

# Create dashboard configuration
dashboard_config = DashboardConfig(
    name="churn_model_monitoring",
    model_name="customer_churn_predictor",
    metrics=["accuracy", "f1_score", "precision", "recall"],
    features=["tenure", "monthly_charges", "total_charges"],
    time_range_days=30,
    refresh_schedule="0 * * * *"  # Hourly
)

# Generate dashboard
dashboard_generator = DashboardGenerator(workspace=workspace)
dashboard_url = dashboard_generator.create_dashboard(dashboard_config)

print(f"Dashboard available at: {dashboard_url}")
```

## Scheduled Monitoring

The framework supports scheduling monitoring jobs:

```python
from databricks_mlops.monitoring import MonitoringScheduler
from databricks_mlops.models.monitoring import ScheduleConfig

# Create schedule configuration
schedule_config = ScheduleConfig(
    cron_expression="0 */6 * * *",  # Every 6 hours
    timezone="UTC",
    start_date="2025-05-13T00:00:00Z",
    pause_status="UNPAUSED"
)

# Schedule monitoring job
scheduler = MonitoringScheduler(workspace=workspace)
job_id = scheduler.schedule_monitoring(
    model_name="customer_churn_predictor",
    config=schedule_config
)

print(f"Scheduled monitoring job with ID: {job_id}")
```

## Custom Monitoring Metrics

The framework supports defining custom metrics:

```python
from databricks_mlops.monitoring import MetricRegistry
from databricks_mlops.models.monitoring import CustomMetric, MetricType

# Define custom metrics
custom_metrics = [
    CustomMetric(
        name="business_impact",
        description="Business impact of model predictions",
        sql_expression="SUM(CASE WHEN prediction = true AND actual = false THEN cost ELSE 0 END) * -1.0",
        type=MetricType.NUMERIC,
        better_is="higher"
    ),
    CustomMetric(
        name="false_positive_rate",
        description="False positive rate",
        sql_expression="SUM(CASE WHEN prediction = true AND actual = false THEN 1 ELSE 0 END) / COUNT(*)",
        type=MetricType.NUMERIC,
        better_is="lower"
    )
]

# Register custom metrics
metric_registry = MetricRegistry(workspace=workspace)
for metric in custom_metrics:
    metric_registry.register_metric(metric)
```

## Best Practices

1. **Set a Baseline**: Always establish a reference dataset that represents the expected distribution
2. **Monitor Feature Importance**: Pay special attention to high-importance features
3. **Define Thresholds Carefully**: Set alert thresholds based on business requirements
4. **Scheduled Monitoring**: Run monitoring jobs on a regular schedule
5. **Automate Actions**: Set up automatic actions for certain drift thresholds
6. **Periodic Retraining**: Establish a process for retraining when drift exceeds thresholds

## Integration with MLflow

The framework integrates with MLflow for tracking monitoring metrics:

```python
from databricks_mlops.monitoring import MLflowMonitoring
from databricks_mlops.utils.auth import WorkspaceConfig

# Create MLflow monitoring
mlflow_monitoring = MLflowMonitoring(
    workspace=WorkspaceConfig(
        host="https://your-workspace.cloud.databricks.com",
        token="${DATABRICKS_TOKEN}"
    )
)

# Log monitoring metrics to MLflow
mlflow_monitoring.log_metrics(
    model_name="customer_churn_predictor",
    metrics={
        "accuracy": 0.92,
        "f1_score": 0.89,
        "drift_score": 0.03
    },
    tags={
        "environment": "production",
        "version": "1.0.0"
    }
)
```

## Troubleshooting

### Common Issues

1. **Missing Reference Data**:
   - Error: `Reference dataset not found`
   - Solution: Ensure the reference dataset path is correct and accessible

2. **Insufficient Permissions**:
   - Error: `Permission denied accessing dataset`
   - Solution: Ensure the access token has proper permissions

3. **Invalid Metric**:
   - Error: `Metric 'xxx' not supported`
   - Solution: Check the list of supported metrics or register a custom metric

4. **Compute Issues**:
   - Error: `Compute resources unavailable`
   - Solution: Check cluster availability or adjust compute configuration

## Next Steps

- See the [Deployment Guide](DEPLOYMENT.md) for model deployment options
- Explore [Model Serving](MODEL_SERVING.md) for accessing deployed models
- Check [Access Bundles](ACCESS_BUNDLES.md) for authentication best practices
