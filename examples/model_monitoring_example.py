#!/usr/bin/env python
"""
Model Monitoring Example

This script demonstrates how to use the monitoring components of the Databricks MLOps
framework to detect data and concept drift, collect performance metrics, and set up
alerts when model performance degrades.

Key components demonstrated:
1. Creating monitoring configuration with strong typing
2. Data drift detection using statistical tests
3. Performance metric collection and analysis
4. Setting up drift-based alerts with thresholds
5. Generating monitoring reports with Pydantic models
"""
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from databricks_mlops.models.base import DriftResult, MetricSeverity, StatusEnum
from databricks_mlops.models.config import MonitoringConfig
from databricks_mlops.monitoring import DriftDetector, MetricCollector
from databricks_mlops.utils.logging import LogLevel, setup_logger


def generate_drift_data(reference_data: pd.DataFrame, drift_factor: float = 0.5, 
                       sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Generate data with controlled drift for monitoring testing.
    
    Args:
        reference_data: Reference data used as baseline
        drift_factor: Amount of drift to introduce (0=no drift, 1=maximum drift)
        sample_size: Optional size to sample, defaults to same size as reference
        
    Returns:
        DataFrame with controlled drift from reference data
    """
    if sample_size is None:
        sample_size = len(reference_data)
    
    # Create a copy of reference data
    drifted_data = reference_data.sample(sample_size, replace=True).copy()
    
    # No drift case
    if drift_factor == 0:
        return drifted_data
    
    # Apply controlled drift to numerical columns
    numerical_columns = drifted_data.select_dtypes(include=['number']).columns
    
    for col in numerical_columns:
        # Skip target column
        if col == 'churned':
            continue
            
        # Calculate shift amount based on column statistics and drift factor
        col_mean = drifted_data[col].mean()
        col_std = drifted_data[col].std()
        
        # Apply shift - larger shift for higher drift factor
        shift = col_std * drift_factor * 2
        drifted_data[col] += shift
        
        # For some columns, also change distribution shape
        if np.random.random() < drift_factor:
            # Apply skew for more severe drift
            drifted_data[col] = np.exp(drifted_data[col] / drifted_data[col].max() * drift_factor) * drifted_data[col]
            # Re-scale to maintain approximate range
            drifted_data[col] = (drifted_data[col] - drifted_data[col].mean()) / drifted_data[col].std() * col_std + col_mean
    
    # Apply controlled drift to categorical columns
    categorical_columns = drifted_data.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_columns:
        # Calculate distribution change probability
        change_prob = drift_factor / 2
        
        # Get value counts of each category
        val_counts = reference_data[col].value_counts(normalize=True)
        
        # Function to potentially change value based on drift factor
        def drift_category(val):
            if np.random.random() < change_prob:
                # Sample a different category with higher probability for less frequent categories
                weights = 1 - val_counts
                weights = weights / weights.sum()
                return np.random.choice(val_counts.index, p=weights)
            return val
        
        # Apply categorical drift
        if change_prob > 0:
            drifted_data[col] = drifted_data[col].apply(drift_category)
    
    # Handle target differently - make predictions harder by changing some labels
    if 'churned' in drifted_data.columns:
        # Flip some labels - more flips for higher drift factor
        flip_mask = np.random.random(len(drifted_data)) < (drift_factor * 0.4)
        if flip_mask.sum() > 0:
            drifted_data.loc[flip_mask, 'churned'] = 1 - drifted_data.loc[flip_mask, 'churned']
    
    return drifted_data


class MonitoringReport(BaseModel):
    """Strongly-typed class for monitoring reports."""
    timestamp: datetime = Field(default_factory=datetime.now)
    model_name: str
    has_data_drift: bool = False
    drift_score: float = 0.0
    drifted_features: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    status: StatusEnum = StatusEnum.SUCCESS
    alerts: List[str] = Field(default_factory=list)
    sample_size: int
    reference_data_size: int
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary with proper datetime handling."""
        result = self.model_dump()
        result["timestamp"] = self.timestamp.isoformat()
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string with proper serialization."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def print_report(self) -> None:
        """Print a formatted monitoring report."""
        alert_emoji = "🚨" if self.has_data_drift else "✅"
        
        print(f"\n{alert_emoji} Monitoring Report for {self.model_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Status: {self.status}")
        print(f"Sample Size: {self.sample_size} (Reference: {self.reference_data_size})")
        
        print("\nDrift Analysis:")
        if self.has_data_drift:
            print(f"  - Data Drift Detected! Overall score: {self.drift_score:.4f}")
            print("  - Drifted Features:")
            for feature in self.drifted_features:
                print(f"      - {feature}")
        else:
            print(f"  - No significant data drift detected. Score: {self.drift_score:.4f}")
        
        print("\nPerformance Metrics:")
        for metric, value in self.performance_metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        if self.alerts:
            print("\nAlerts:")
            for alert in self.alerts:
                print(f"  - {alert}")


def run_monitoring_example() -> None:
    """Run the model monitoring example."""
    # Set up logging
    logger = setup_logger("model_monitoring", LogLevel.INFO)
    logger.info("Starting model monitoring example")
    
    # Step 1: Create or load reference data (same as in the churn prediction example)
    from customer_churn_prediction import load_sample_data
    reference_data = load_sample_data()
    logger.info(f"Loaded reference data with {len(reference_data)} records")
    
    # Step 2: Create monitoring configuration with strong typing
    monitoring_config = MonitoringConfig(
        model_name="customer_churn_predictor",
        endpoint_name="customer-churn-endpoint",
        metrics=["accuracy", "f1_score", "drift_score", "data_quality"],
        monitor_data_drift=True,
        monitor_prediction_drift=True,
        reference_dataset_path="reference_data",  # In practice, this would be a Delta table path
        alert_thresholds={
            "drift_score": 0.05,            # Alert if drift score > 0.05
            "accuracy_drop": 0.1,           # Alert if accuracy drops by 10%
            "prediction_drift_score": 0.1,  # Alert if prediction distribution changes significantly
            "data_quality_score": 0.8       # Alert if data quality falls below 80%
        },
        monitoring_schedule="0 */6 * * *",  # Every 6 hours (in production)
        lookback_days=7
    )
    
    # Step 3: Initialize drift detector and metric collector
    drift_detector = DriftDetector(config=monitoring_config)
    metric_collector = MetricCollector(config=monitoring_config)
    
    # Step 4: Run monitoring for different scenarios
    drift_scenarios = [
        {"name": "No Drift", "factor": 0.0},
        {"name": "Low Drift", "factor": 0.2},
        {"name": "Medium Drift", "factor": 0.5},
        {"name": "High Drift", "factor": 0.8}
    ]
    
    # Prepare reference data - remove ID and timestamp columns
    reference_features = reference_data.drop(['customer_id', 'last_update_time'], axis=1)
    
    # Track all reports for final summary
    all_reports = []
    
    # Monitor each scenario
    for scenario in drift_scenarios:
        logger.info(f"Running monitoring for scenario: {scenario['name']}")
        
        # Generate data with specified drift level
        current_data = generate_drift_data(
            reference_data, 
            drift_factor=scenario['factor'],
            sample_size=1000
        )
        
        # Prepare current data - remove ID and timestamp columns
        current_features = current_data.drop(['customer_id', 'last_update_time'], axis=1)
        
        # Split into features and target
        current_X = current_features.drop('churned', axis=1)
        current_y = current_features['churned']
        
        # Create actual predictions (in a real scenario, these would come from the deployed model)
        # Here we simulate predictions with controlled quality based on drift factor
        accuracy_drop = scenario['factor'] * 0.3  # Higher drift factor = lower accuracy
        base_accuracy = 0.9  # Start with 90% accuracy
        targeted_accuracy = base_accuracy - accuracy_drop
        
        # Generate predictions with controlled accuracy
        correct_predictions = np.random.choice(
            [True, False], 
            size=len(current_y), 
            p=[targeted_accuracy, 1-targeted_accuracy]
        )
        y_pred = current_y.copy()
        flip_mask = ~correct_predictions
        if flip_mask.sum() > 0:
            y_pred.loc[flip_mask] = 1 - y_pred.loc[flip_mask]
        
        # Run drift detection
        drift_result = drift_detector.detect_drift(
            reference_data=reference_features,
            current_data=current_features
        )
        
        # Collect performance metrics
        metrics_result = metric_collector.collect_metrics(
            y_true=current_y,
            y_pred=y_pred,
            timestamps=pd.Series([datetime.now()] * len(current_y))
        )
        
        # Create monitoring report
        alerts = []
        if drift_result.has_drift:
            alerts.append(f"Data drift detected with score {drift_result.drift_score:.4f}")
        
        if metrics_result.metrics.get('accuracy', 1.0) < 0.7:
            alerts.append(f"Model accuracy below threshold: {metrics_result.metrics.get('accuracy'):.4f}")
            
        report = MonitoringReport(
            model_name="customer_churn_predictor",
            has_data_drift=drift_result.has_drift,
            drift_score=drift_result.drift_score,
            drifted_features=drift_result.drifted_features,
            performance_metrics=metrics_result.metrics,
            status=StatusEnum.WARNING if alerts else StatusEnum.SUCCESS,
            alerts=alerts,
            sample_size=len(current_features),
            reference_data_size=len(reference_features)
        )
        
        # Display the report
        report.print_report()
        all_reports.append(report)
        
        # In a real scenario, this would be stored in a database or dashboard
        logger.info(f"Completed monitoring for scenario: {scenario['name']}")
    
    # Print summary comparison
    print("\n📊 Monitoring Results Comparison")
    print("-------------------------------")
    print(f"{'Scenario':<12} | {'Drift Score':<12} | {'Accuracy':<10} | {'Status':<8}")
    print("-------------------------------")
    for scenario, report in zip(drift_scenarios, all_reports):
        accuracy = report.performance_metrics.get('accuracy', 0.0)
        status = "⚠️ ALERT" if report.alerts else "✅ OK"
        print(f"{scenario['name']:<12} | {report.drift_score:<12.4f} | {accuracy:<10.4f} | {status:<8}")


if __name__ == "__main__":
    try:
        run_monitoring_example()
    except ValidationError as e:
        print(f"Validation error in configuration: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running monitoring example: {e}")
        sys.exit(1)
