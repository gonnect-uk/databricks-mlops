#!/usr/bin/env python
"""
Complete MLOps pipeline example using Databricks MLOps framework.

This script demonstrates an end-to-end ML pipeline with strong typing and Pydantic models
throughout each stage:
1. Data validation and processing
2. Feature engineering 
3. Model training
4. Model deployment
5. Model monitoring

Run this script as:
    python complete_mlops_pipeline.py --config-dir /path/to/configs
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from databricks_mlops.config import MLOpsConfigManager
from databricks_mlops.core import (DataPipeline, DeploymentPipeline, FeaturePipeline,
                                 MonitoringPipeline, PipelineOrchestrator, 
                                 PipelineStage, PipelineType, TrainingPipeline)
from databricks_mlops.models.base import Result, StatusEnum
from databricks_mlops.models.config import (DataConfig, DeploymentConfig, FeatureConfig,
                                          ModelConfig, MonitoringConfig, PipelineConfig)
from databricks_mlops.monitoring import DriftDetector, MetricCollector
from databricks_mlops.pipelines import (FeatureTransformer, ModelDeployer, ModelTrainer,
                                       TrainingConfig)
from databricks_mlops.utils.databricks_client import DatabricksConfig
from databricks_mlops.utils.logging import LogLevel, setup_logger
from databricks_mlops.workflows.mlflow_tracking import ModelStage, TrackingConfig


class PipelineRunResult(BaseModel):
    """Strongly-typed result of a complete pipeline run."""
    pipeline_name: str
    status: StatusEnum
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    execution_time_seconds: float = 0.0
    stage_results: Dict[str, Result] = Field(default_factory=dict)
    failed_stages: List[str] = Field(default_factory=list)
    artifact_uris: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Return True if the pipeline run was successful."""
        return self.status == StatusEnum.SUCCESS


class MLOpsPipelineRunner:
    """
    Orchestrates a complete MLOps pipeline using the Databricks MLOps framework.
    
    This class demonstrates how to use the framework's components together 
    to create an end-to-end ML pipeline with strong type safety throughout.
    """
    
    def __init__(self, config_dir: str):
        """
        Initialize the MLOps pipeline runner.
        
        Args:
            config_dir: Directory containing configuration YAML files
        """
        self.config_dir = Path(config_dir)
        self.logger = setup_logger("mlops_pipeline", LogLevel.INFO)
        
        # Validate config directory exists
        if not self.config_dir.exists():
            raise ValueError(f"Configuration directory {self.config_dir} does not exist")
        
        # Initialize the orchestrator
        self.orchestrator = PipelineOrchestrator()

    def _load_config(self, config_file: str, config_class: Type[BaseModel]) -> Any:
        """
        Load a configuration from a YAML file with strong typing.
        
        Args:
            config_file: Configuration file name
            config_class: Pydantic model class for the configuration
            
        Returns:
            Loaded and validated configuration
        """
        config_path = self.config_dir / config_file
        if not config_path.exists():
            raise ValueError(f"Configuration file {config_path} does not exist")
        
        # Use MLOpsConfigManager for typed configuration loading
        config_manager = MLOpsConfigManager.create_pipeline_config_manager()
        return config_manager.load_from_yaml(str(config_path))

    def setup_data_pipeline(self) -> None:
        """Set up the data processing pipeline stage."""
        # Load configuration with strong typing
        config = self._load_config("data_config.yaml", PipelineConfig)
        
        # Validate configuration structure
        if not config.data_config:
            raise ValueError("Data configuration not found in data_config.yaml")
        
        # Add data pipeline stage to orchestrator
        self.orchestrator.add_data_stage(
            name="data_processing",
            config=config,
            enabled=True
        )
        
        self.logger.info(f"Added data pipeline stage: {config.name}")

    def setup_feature_pipeline(self) -> None:
        """Set up the feature engineering pipeline stage."""
        # Load configuration with strong typing
        config = self._load_config("feature_config.yaml", PipelineConfig)
        
        # Validate configuration structure
        if not config.feature_config:
            raise ValueError("Feature configuration not found in feature_config.yaml")
        
        # Add feature pipeline stage to orchestrator
        self.orchestrator.add_feature_stage(
            name="feature_engineering",
            config=config,
            depends_on=["data_processing"],
            enabled=True
        )
        
        self.logger.info(f"Added feature pipeline stage: {config.name}")

    def setup_training_pipeline(self) -> None:
        """Set up the model training pipeline stage."""
        # Load configuration with strong typing
        config = self._load_config("model_config.yaml", PipelineConfig)
        
        # Validate configuration structure
        if not config.model_config:
            raise ValueError("Model configuration not found in model_config.yaml")
        
        # Add training pipeline stage to orchestrator
        self.orchestrator.add_training_stage(
            name="model_training",
            config=config,
            depends_on=["feature_engineering"],
            enabled=True
        )
        
        self.logger.info(f"Added training pipeline stage: {config.name}")

    def setup_deployment_pipeline(self) -> None:
        """Set up the model deployment pipeline stage."""
        # Load configuration with strong typing
        config = self._load_config("deployment_config.yaml", PipelineConfig)
        
        # Validate configuration structure
        if not config.deployment_config:
            raise ValueError("Deployment configuration not found in deployment_config.yaml")
        
        # Add deployment pipeline stage to orchestrator
        self.orchestrator.add_deployment_stage(
            name="model_deployment",
            config=config,
            depends_on=["model_training"],
            enabled=True
        )
        
        self.logger.info(f"Added deployment pipeline stage: {config.name}")

    def setup_monitoring_pipeline(self) -> None:
        """Set up the model monitoring pipeline stage."""
        # For this example, we'll create a monitoring config programmatically,
        # demonstrating how to build configurations with Pydantic models
        
        # Create a monitoring configuration with strong typing
        monitoring_config = MonitoringConfig(
            model_name="customer_churn_predictor",
            endpoint_name="customer-churn-predictor",
            metrics=["accuracy", "drift_score", "data_quality"],
            monitor_data_drift=True,
            monitor_prediction_drift=True,
            reference_dataset_path="dbfs:/mnt/gold/feature_store/customer_churn_features/reference",
            alert_thresholds={
                "accuracy_drop": 0.05,
                "drift_score": 0.2,
                "data_quality_score": 0.8
            },
            monitoring_schedule="0 */6 * * *",  # Every 6 hours
            lookback_days=7,
            alert_emails=["alerts@example.com"]
        )
        
        # Create a pipeline configuration with the monitoring config
        config = PipelineConfig(
            name="customer_churn_monitoring_pipeline",
            description="Model monitoring pipeline for customer churn prediction",
            owner="mlops_engineer@example.com",
            tags={
                "domain": "customer_analytics",
                "project": "churn_prediction",
                "environment": "staging"
            },
            timeout_minutes=30,
            retry_attempts=3,
            environment="staging",
            monitoring_config=monitoring_config
        )
        
        # Add monitoring pipeline stage to orchestrator
        self.orchestrator.add_monitoring_stage(
            name="model_monitoring",
            config=config,
            depends_on=["model_deployment"],
            enabled=True
        )
        
        self.logger.info(f"Added monitoring pipeline stage: {config.name}")

    def setup_pipeline(self) -> None:
        """Set up the complete MLOps pipeline with all stages."""
        self.setup_data_pipeline()
        self.setup_feature_pipeline()
        self.setup_training_pipeline()
        self.setup_deployment_pipeline()
        self.setup_monitoring_pipeline()

    def run_pipeline(self) -> PipelineRunResult:
        """
        Run the complete MLOps pipeline.
        
        Returns:
            PipelineRunResult: Strongly-typed result of the pipeline run
        """
        self.logger.info("Starting complete MLOps pipeline run")
        start_time = datetime.now()
        
        # Run the orchestrated pipeline
        result = self.orchestrator.run()
        
        # Convert to our strongly-typed result model
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        pipeline_result = PipelineRunResult(
            pipeline_name="customer_churn_mlops_pipeline",
            status=result.status,
            start_time=start_time,
            end_time=end_time,
            execution_time_seconds=execution_time,
            stage_results=result.stage_results,
            failed_stages=result.failed_stages
        )
        
        # Extract important metrics and artifacts
        for stage_name, stage_result in result.stage_results.items():
            if hasattr(stage_result, "model_uri") and stage_result.model_uri:
                pipeline_result.artifact_uris[f"{stage_name}_model"] = stage_result.model_uri
                
            if hasattr(stage_result, "evaluation") and hasattr(stage_result.evaluation, "metrics"):
                for metric_name, metric_value in stage_result.evaluation.metrics.items():
                    pipeline_result.metrics[f"{stage_name}_{metric_name}"] = metric_value
        
        # Log summary
        if pipeline_result.is_success:
            self.logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
        else:
            self.logger.error(f"Pipeline failed in {execution_time:.2f} seconds. Failed stages: {', '.join(result.failed_stages)}")
        
        return pipeline_result


def main() -> None:
    """Main entry point for the MLOps pipeline example."""
    parser = argparse.ArgumentParser(description="Run a complete MLOps pipeline")
    parser.add_argument("--config-dir", required=True, help="Directory containing configuration YAML files")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                     help="Logging level")
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(LogLevel, args.log_level)
    logger = setup_logger("main", log_level)
    
    try:
        # Create and run the pipeline
        runner = MLOpsPipelineRunner(args.config_dir)
        runner.setup_pipeline()
        result = runner.run_pipeline()
        
        # Output summary
        success_emoji = "✅" if result.is_success else "❌"
        print(f"\n{success_emoji} Pipeline run complete")
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_seconds:.2f} seconds")
        print(f"Start time: {result.start_time}")
        print(f"End time: {result.end_time}")
        
        if result.metrics:
            print("\nMetrics:")
            for metric_name, metric_value in result.metrics.items():
                print(f"  - {metric_name}: {metric_value:.4f}")
        
        if result.artifact_uris:
            print("\nArtifacts:")
            for artifact_name, artifact_uri in result.artifact_uris.items():
                print(f"  - {artifact_name}: {artifact_uri}")
        
        if result.failed_stages:
            print("\nFailed stages:")
            for stage in result.failed_stages:
                print(f"  - {stage}")
        
        # Exit with appropriate code
        sys.exit(0 if result.is_success else 1)
        
    except Exception as e:
        logger.exception(f"Error running MLOps pipeline: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
