"""
Metric collection system for model monitoring with strong typing.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from databricks_mlops.models.base import Metric, Result, StatusEnum
from databricks_mlops.models.config import MonitoringConfig
from databricks_mlops.utils.logging import setup_logger

# Set up logger
logger = setup_logger("metric_collector")


class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC = "auc"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2 = "r2"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    PREDICTION_DRIFT = "prediction_drift"
    DATA_DRIFT = "data_drift"
    FEATURE_IMPORTANCE = "feature_importance"
    CUSTOM = "custom"


class CollectionFrequency(str, Enum):
    """Frequency at which metrics are collected."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ON_DEMAND = "on_demand"
    CONTINUOUS = "continuous"


class MetricCollectionConfig(BaseModel):
    """Configuration for metric collection."""
    metric_types: List[MetricType]
    frequency: CollectionFrequency
    model_name: str
    endpoint_name: Optional[str] = None
    reference_dataset_path: Optional[str] = None
    current_dataset_path: Optional[str] = None
    lookback_days: int = 7
    aggregation_window: str = "1d"  # Pandas-compatible freq string ('1d', '1h', etc.)
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_dataset_paths(self) -> 'MetricCollectionConfig':
        """Validate dataset paths are provided if needed for drift detection."""
        if (MetricType.DATA_DRIFT in self.metric_types or 
            MetricType.PREDICTION_DRIFT in self.metric_types) and not self.reference_dataset_path:
            raise ValueError("reference_dataset_path is required for drift detection metrics")
        return self


class MetricCollectionResult(Result):
    """Result of metric collection."""
    model_name: str
    endpoint_name: Optional[str] = None
    collection_time: datetime = Field(default_factory=datetime.now)
    metrics: List[Metric] = Field(default_factory=list)
    metric_count: int = 0


class MetricCollector:
    """
    Collects and processes metrics for model monitoring.
    
    This class provides methods to collect various types of metrics
    for model performance and data/prediction drift detection.
    """
    
    def __init__(self, config: MetricCollectionConfig):
        """
        Initialize the metric collector.
        
        Args:
            config: Metric collection configuration
        """
        self.config = config
        self.logger = logger
    
    def collect_metrics(self) -> MetricCollectionResult:
        """
        Collect metrics according to the configuration.
        
        Returns:
            MetricCollectionResult: The result of metric collection
        """
        self.logger.info(f"Collecting metrics for model {self.config.model_name}")
        
        # Initialize result
        result = MetricCollectionResult(
            status=StatusEnum.SUCCESS,
            message=f"Metric collection started for model {self.config.model_name}",
            model_name=self.config.model_name,
            endpoint_name=self.config.endpoint_name
        )
        
        collected_metrics = []
        
        try:
            # Collect each requested metric type
            for metric_type in self.config.metric_types:
                self.logger.info(f"Collecting {metric_type} metrics")
                
                if metric_type == MetricType.DATA_DRIFT:
                    metrics = self._collect_data_drift_metrics()
                elif metric_type == MetricType.PREDICTION_DRIFT:
                    metrics = self._collect_prediction_drift_metrics()
                elif metric_type in [MetricType.ACCURACY, MetricType.PRECISION, 
                                    MetricType.RECALL, MetricType.F1_SCORE, MetricType.AUC]:
                    metrics = self._collect_classification_metrics(metric_type)
                elif metric_type in [MetricType.MAE, MetricType.MSE, MetricType.RMSE, MetricType.R2]:
                    metrics = self._collect_regression_metrics(metric_type)
                elif metric_type in [MetricType.LATENCY, MetricType.THROUGHPUT]:
                    metrics = self._collect_performance_metrics(metric_type)
                elif metric_type == MetricType.FEATURE_IMPORTANCE:
                    metrics = self._collect_feature_importance()
                elif metric_type == MetricType.CUSTOM:
                    metrics = self._collect_custom_metrics()
                else:
                    self.logger.warning(f"Unsupported metric type: {metric_type}")
                    metrics = []
                
                collected_metrics.extend(metrics)
            
            # Update the result
            result.metrics = collected_metrics
            result.metric_count = len(collected_metrics)
            result.message = f"Successfully collected {len(collected_metrics)} metrics for model {self.config.model_name}"
            
            self.logger.info(f"Collected {len(collected_metrics)} metrics for model {self.config.model_name}")
            return result
            
        except Exception as e:
            error_msg = f"Error collecting metrics: {str(e)}"
            self.logger.exception(error_msg)
            
            result.status = StatusEnum.FAILED
            result.message = error_msg
            result.errors = [{"type": type(e).__name__, "message": str(e)}]
            
            return result
    
    def _collect_data_drift_metrics(self) -> List[Metric]:
        """
        Collect data drift metrics.
        
        Returns:
            List of collected metrics
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Load reference and current datasets
        # 2. Calculate distribution differences
        # 3. Generate drift metrics
        
        self.logger.info("Data drift detection not fully implemented")
        
        # Example metric
        return [
            Metric(
                name="data_drift_score",
                value=0.05,  # Example value
                labels={
                    "model": self.config.model_name,
                    "type": "data_drift"
                }
            )
        ]
    
    def _collect_prediction_drift_metrics(self) -> List[Metric]:
        """
        Collect prediction drift metrics.
        
        Returns:
            List of collected metrics
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Get predictions on reference and current datasets
        # 2. Compare prediction distributions
        # 3. Generate drift metrics
        
        self.logger.info("Prediction drift detection not fully implemented")
        
        # Example metric
        return [
            Metric(
                name="prediction_drift_score",
                value=0.03,  # Example value
                labels={
                    "model": self.config.model_name,
                    "type": "prediction_drift"
                }
            )
        ]
    
    def _collect_classification_metrics(self, metric_type: MetricType) -> List[Metric]:
        """
        Collect classification model performance metrics.
        
        Args:
            metric_type: Type of classification metric to collect
            
        Returns:
            List of collected metrics
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Get model predictions
        # 2. Calculate classification metrics
        
        self.logger.info(f"Collecting {metric_type} classification metrics")
        
        # Example metric
        return [
            Metric(
                name=metric_type,
                value=0.92,  # Example value
                labels={
                    "model": self.config.model_name,
                    "type": "classification_performance"
                }
            )
        ]
    
    def _collect_regression_metrics(self, metric_type: MetricType) -> List[Metric]:
        """
        Collect regression model performance metrics.
        
        Args:
            metric_type: Type of regression metric to collect
            
        Returns:
            List of collected metrics
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Get model predictions
        # 2. Calculate regression metrics
        
        self.logger.info(f"Collecting {metric_type} regression metrics")
        
        # Example metrics
        if metric_type == MetricType.MAE:
            value = 0.25  # Example value
        elif metric_type == MetricType.MSE:
            value = 0.12  # Example value
        elif metric_type == MetricType.RMSE:
            value = 0.35  # Example value
        elif metric_type == MetricType.R2:
            value = 0.87  # Example value
        else:
            value = 0.0
        
        return [
            Metric(
                name=metric_type,
                value=value,
                labels={
                    "model": self.config.model_name,
                    "type": "regression_performance"
                }
            )
        ]
    
    def _collect_performance_metrics(self, metric_type: MetricType) -> List[Metric]:
        """
        Collect model serving performance metrics.
        
        Args:
            metric_type: Type of performance metric to collect
            
        Returns:
            List of collected metrics
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Query the model endpoint
        # 2. Collect latency/throughput metrics
        
        self.logger.info(f"Collecting {metric_type} performance metrics")
        
        # Example metrics
        if metric_type == MetricType.LATENCY:
            value = 120.5  # Example value in ms
            name = "p95_latency_ms"
        elif metric_type == MetricType.THROUGHPUT:
            value = 250.0  # Example value in requests/sec
            name = "throughput_rps"
        else:
            value = 0.0
            name = metric_type
        
        return [
            Metric(
                name=name,
                value=value,
                labels={
                    "model": self.config.model_name,
                    "endpoint": self.config.endpoint_name or "unknown",
                    "type": "performance"
                }
            )
        ]
    
    def _collect_feature_importance(self) -> List[Metric]:
        """
        Collect feature importance metrics.
        
        Returns:
            List of collected metrics
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Load the model
        # 2. Extract feature importance scores
        
        self.logger.info("Collecting feature importance metrics")
        
        # Example features
        features = {
            "feature1": 0.35,
            "feature2": 0.25,
            "feature3": 0.20,
            "feature4": 0.15,
            "feature5": 0.05
        }
        
        # Create a metric for each feature
        return [
            Metric(
                name=f"feature_importance_{feature}",
                value=importance,
                labels={
                    "model": self.config.model_name,
                    "feature": feature,
                    "type": "feature_importance"
                }
            )
            for feature, importance in features.items()
        ]
    
    def _collect_custom_metrics(self) -> List[Metric]:
        """
        Collect custom metrics defined in the configuration.
        
        Returns:
            List of collected metrics
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Parse custom metric definitions
        # 2. Execute custom logic
        
        self.logger.info("Collecting custom metrics")
        
        # Use custom metrics from config
        return [
            Metric(
                name=name,
                value=value,
                labels={
                    "model": self.config.model_name,
                    "type": "custom"
                }
            )
            for name, value in self.config.custom_metrics.items()
        ]
