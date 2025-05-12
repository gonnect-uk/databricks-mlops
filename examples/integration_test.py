#!/usr/bin/env python
"""
Integration Test for Databricks MLOps Framework

This script demonstrates how the different components of the MLOps framework
work together in an integrated workflow. It tests a complete ML pipeline from
data validation through model deployment and monitoring.

The test uses Pydantic models for strong typing throughout the entire process
and showcases error handling, validation, and the correct sequence of operations.
"""
import logging
import os
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import mlflow
import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, Field, ValidationError

from databricks_mlops.config import MLOpsConfigManager
from databricks_mlops.core import (DataPipeline, DeploymentPipeline, FeaturePipeline,
                                 MonitoringPipeline, PipelineOrchestrator, 
                                 PipelineStage, PipelineType, TrainingPipeline)
from databricks_mlops.models.base import Result, StatusEnum, ValidationResult
from databricks_mlops.models.config import (DataConfig, DeploymentConfig, FeatureConfig,
                                          ModelConfig, MonitoringConfig, PipelineConfig)
from databricks_mlops.monitoring import DriftDetector, MetricCollector
from databricks_mlops.pipelines import (FeatureScope, FeatureTransformer, ModelDeployer, 
                                       ModelFramework, ModelTrainer, ModelType, ScalerType, 
                                       TrainingConfig)
from databricks_mlops.utils.data_validation import DataValidator, ValidationRule
from databricks_mlops.utils.logging import LogLevel, setup_logger


class TestIntegration(unittest.TestCase):
    """Integration test suite for the Databricks MLOps framework."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test environment once for all test methods."""
        # Set up logging
        cls.logger = setup_logger("integration_test", LogLevel.INFO)
        cls.logger.info("Setting up integration test suite")
        
        # Create a temporary directory for artifacts
        cls.temp_dir = Path("./temp_test_artifacts")
        cls.temp_dir.mkdir(exist_ok=True)
        
        # Set up MLflow for local tracking
        mlflow.set_tracking_uri(f"file://{cls.temp_dir}/mlruns")
        mlflow.set_experiment("integration_test")
        
        # Load sample data
        from customer_churn_prediction import load_sample_data
        cls.data = load_sample_data()
        cls.logger.info(f"Loaded sample data with {len(cls.data)} records")
        
        # Split data for different stages
        cls.train_data = cls.data.sample(frac=0.6, random_state=42)
        remaining = cls.data.drop(cls.train_data.index)
        cls.val_data = remaining.sample(frac=0.5, random_state=42)
        cls.test_data = remaining.drop(cls.val_data.index)
        
        # Save the datasets for reference
        cls.train_data.to_csv(f"{cls.temp_dir}/train_data.csv", index=False)
        cls.val_data.to_csv(f"{cls.temp_dir}/val_data.csv", index=False)
        cls.test_data.to_csv(f"{cls.temp_dir}/test_data.csv", index=False)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up after all tests have run."""
        # Clean up created files (uncomment for actual cleanup)
        # import shutil
        # shutil.rmtree(cls.temp_dir)
        cls.logger.info("Integration test suite tear down complete")

    def test_01_data_validation(self) -> None:
        """Test data validation component with strong typing."""
        self.logger.info("Testing data validation component")
        
        # Define validation rules using strongly-typed models
        rules = [
            ValidationRule(
                name="customer_id_not_null",
                condition="customer_id is not null",
                severity="error",
                description="Customer ID should never be null"
            ),
            ValidationRule(
                name="positive_tenure",
                condition="tenure >= 0",
                severity="error",
                description="Tenure cannot be negative"
            ),
            ValidationRule(
                name="valid_charges",
                condition="monthly_charges > 0 and total_charges >= monthly_charges",
                severity="warning",
                description="Charges should be positive and total >= monthly"
            )
        ]
        
        # Create data validator
        validator = DataValidator(rules=rules)
        
        # Validate the data
        result = validator.validate(self.train_data)
        
        # Assertions with strong typing
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.status, StatusEnum.SUCCESS)
        self.assertEqual(len(result.failed_rules), 0)
        
        # Test with invalid data
        invalid_data = self.train_data.copy()
        invalid_data.loc[0, 'tenure'] = -1  # Create a rule violation
        
        result = validator.validate(invalid_data)
        self.assertEqual(result.status, StatusEnum.ERROR)
        self.assertEqual(len(result.failed_rules), 1)
        self.assertEqual(result.failed_rules[0].name, "positive_tenure")

    def test_02_feature_engineering(self) -> None:
        """Test feature engineering pipeline with proper typing."""
        self.logger.info("Testing feature engineering pipeline")
        
        # Create feature engineering configuration with strong typing
        config = FeatureConfig(
            categorical_features=[
                'contract_type', 'payment_method', 'subscription_type',
                'online_security', 'tech_support', 'streaming_tv',
                'streaming_movies', 'gender', 'partner', 'dependents'
            ],
            numerical_features=[
                'tenure', 'monthly_charges', 'total_charges', 'senior_citizen'
            ],
            target_column='churned',
            transformers=[
                {
                    'name': 'numeric_scaler',
                    'type': ScalerType.STANDARD,
                    'features': ['tenure', 'monthly_charges', 'total_charges'],
                    'scope': FeatureScope.NUMERICAL
                },
                {
                    'name': 'categorical_encoder',
                    'type': 'one_hot',
                    'features': [
                        'contract_type', 'payment_method', 'subscription_type',
                        'online_security', 'tech_support', 'streaming_tv',
                        'streaming_movies', 'gender', 'partner', 'dependents'
                    ],
                    'scope': FeatureScope.CATEGORICAL
                }
            ]
        )
        
        # Create feature transformer
        transformer = FeatureTransformer(config=config)
        
        # Fit and transform
        train_features = self.train_data.drop(['customer_id', 'last_update_time'], axis=1)
        transformed_data = transformer.fit_transform(train_features)
        
        # Validate transformation results
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertGreater(len(transformed_data.columns), len(train_features.columns))
        
        # Validate encoding results - one-hot columns should exist
        self.assertIn('contract_type_Month-to-month', transformed_data.columns)
        self.assertIn('payment_method_Electronic check', transformed_data.columns)
        
        # Validate scaling results - numeric columns should have mean close to 0
        self.assertAlmostEqual(transformed_data['tenure'].mean(), 0, delta=0.1)
        self.assertAlmostEqual(transformed_data['monthly_charges'].mean(), 0, delta=0.1)
        
        # Test transformation on validation data
        val_features = self.val_data.drop(['customer_id', 'last_update_time'], axis=1)
        val_transformed = transformer.transform(val_features)
        
        # Validate val transformation has same structure
        self.assertEqual(transformed_data.columns.tolist(), val_transformed.columns.tolist())

    def test_03_model_training(self) -> None:
        """Test model training pipeline with strong typing."""
        self.logger.info("Testing model training pipeline")
        
        # Set up feature engineering
        feature_config = FeatureConfig(
            categorical_features=[
                'contract_type', 'payment_method', 'subscription_type',
                'online_security', 'tech_support', 'streaming_tv', 
                'streaming_movies', 'gender', 'partner', 'dependents'
            ],
            numerical_features=[
                'tenure', 'monthly_charges', 'total_charges', 'senior_citizen'
            ],
            target_column='churned',
            transformers=[
                {
                    'name': 'numeric_scaler',
                    'type': ScalerType.STANDARD,
                    'features': ['tenure', 'monthly_charges', 'total_charges'],
                    'scope': FeatureScope.NUMERICAL
                },
                {
                    'name': 'categorical_encoder',
                    'type': 'one_hot',
                    'features': [
                        'contract_type', 'payment_method', 'subscription_type',
                        'online_security', 'tech_support', 'streaming_tv',
                        'streaming_movies', 'gender', 'partner', 'dependents'
                    ],
                    'scope': FeatureScope.CATEGORICAL
                }
            ]
        )
        
        # Create feature transformer
        transformer = FeatureTransformer(config=feature_config)
        
        # Prepare data
        train_features = self.train_data.drop(['customer_id', 'last_update_time'], axis=1)
        val_features = self.val_data.drop(['customer_id', 'last_update_time'], axis=1)
        
        # Transform data
        train_transformed = transformer.fit_transform(train_features)
        val_transformed = transformer.transform(val_features)
        
        # Create training configuration with strong typing
        training_config = TrainingConfig(
            model_name="integration_test_model",
            model_type=ModelType.CLASSIFICATION,
            framework=ModelFramework.SKLEARN,
            hyperparameters={
                'n_estimators': 50,
                'max_depth': 5,
                'min_samples_split': 5,
                'random_state': 42
            },
            metrics=["accuracy", "precision", "recall", "f1"],
            experiment_name="integration_test"
        )
        
        # Train model using ModelTrainer
        X_train = train_transformed.drop('churned', axis=1)
        y_train = train_transformed['churned']
        X_val = val_transformed.drop('churned', axis=1)
        y_val = val_transformed['churned']
        
        # Initialize model trainer
        trainer = ModelTrainer(config=training_config)
        
        # Train and log model
        with mlflow.start_run(run_name="integration_test_run") as run:
            result = trainer.train_and_evaluate(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                register_model=True
            )
        
        # Validate training results
        self.assertIsInstance(result, Result)
        self.assertEqual(result.status, StatusEnum.SUCCESS)
        self.assertIsNotNone(result.model)
        self.assertIsNotNone(result.model_uri)
        
        # Validate that metrics were captured
        self.assertIn('accuracy', result.metrics)
        self.assertIn('precision', result.metrics)
        self.assertIn('recall', result.metrics)
        self.assertIn('f1', result.metrics)
        
        # Store model URI for later use in deployment
        self.model_uri = result.model_uri

    def test_04_model_deployment(self) -> None:
        """Test model deployment pipeline with strong typing."""
        self.logger.info("Testing model deployment pipeline")
        
        # Skip if model URI not defined (previous test failed)
        if not hasattr(self, 'model_uri'):
            self.skipTest("Model URI not available, skipping deployment test")
        
        # Set up deployment configuration
        deployment_config = DeploymentConfig(
            model_name="integration_test_model",
            model_version="1",
            endpoint_name="integration-test-endpoint",
            environment="test",
            compute_type="cpu",
            deployment_type="batch"  # Use batch instead of serving for easier testing
        )
        
        # Initialize model deployer
        deployer = ModelDeployer(config=deployment_config)
        
        # In a real environment, we would deploy to Databricks
        # For testing, we'll load the model and perform a mock deployment
        model = mlflow.pyfunc.load_model(self.model_uri)
        
        # Mock the deployment (in a real scenario, this would call Databricks APIs)
        endpoint_url = f"https://example.org/model-endpoints/{deployment_config.endpoint_name}"
        mock_result = Result(
            status=StatusEnum.SUCCESS,
            message="Model deployed successfully (mock)",
            details={
                "endpoint_name": deployment_config.endpoint_name,
                "model_uri": self.model_uri,
                "endpoint_url": endpoint_url,
                "deployment_time": datetime.now().isoformat()
            }
        )
        
        # Verify the model can make predictions
        test_features = self.test_data.drop(['customer_id', 'last_update_time', 'churned'], axis=1)
        predictions = model.predict(test_features)
        
        # Validate predictions
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.test_data))
        
        # Only in a real environment would we do an actual deployment
        # self.assertEqual(deployment_result.status, StatusEnum.SUCCESS)

    def test_05_drift_detection(self) -> None:
        """Test drift detection with strong typing."""
        self.logger.info("Testing drift detection")
        
        # Create monitoring configuration
        monitoring_config = MonitoringConfig(
            model_name="integration_test_model",
            endpoint_name="integration-test-endpoint",
            metrics=["accuracy", "drift_score"],
            monitor_data_drift=True,
            reference_dataset_path=f"{self.temp_dir}/train_data.csv",
            alert_thresholds={
                "drift_score": 0.05,
                "accuracy_drop": 0.1
            }
        )
        
        # Create drift detector
        drift_detector = DriftDetector(config=monitoring_config)
        
        # Test with no-drift data (same distribution)
        reference_data = self.train_data.drop(['customer_id', 'last_update_time'], axis=1)
        current_data = self.val_data.drop(['customer_id', 'last_update_time'], axis=1)
        
        no_drift_result = drift_detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data
        )
        
        # Validate result
        self.assertIsInstance(no_drift_result, Result)
        
        # Test with drifted data (modified distribution)
        from model_monitoring_example import generate_drift_data
        
        drifted_data = generate_drift_data(
            reference_data=self.train_data,
            drift_factor=0.8
        ).drop(['customer_id', 'last_update_time'], axis=1)
        
        drift_result = drift_detector.detect_drift(
            reference_data=reference_data,
            current_data=drifted_data
        )
        
        # Validate drift result
        self.assertIsInstance(drift_result, Result)

    def test_06_full_pipeline_orchestration(self) -> None:
        """Test full pipeline orchestration with proper typing."""
        self.logger.info("Testing full pipeline orchestration")
        
        # Create orchestrator
        orchestrator = PipelineOrchestrator()
        
        # Create feature engineering stage
        feature_config = PipelineConfig(
            name="feature_engineering_stage",
            description="Feature engineering stage",
            feature_config=FeatureConfig(
                categorical_features=[
                    'contract_type', 'payment_method', 'subscription_type',
                    'online_security', 'tech_support', 'streaming_tv', 
                    'streaming_movies', 'gender', 'partner', 'dependents'
                ],
                numerical_features=[
                    'tenure', 'monthly_charges', 'total_charges', 'senior_citizen'
                ],
                target_column='churned',
                transformers=[
                    {
                        'name': 'numeric_scaler',
                        'type': ScalerType.STANDARD,
                        'features': ['tenure', 'monthly_charges', 'total_charges'],
                        'scope': FeatureScope.NUMERICAL
                    },
                    {
                        'name': 'categorical_encoder',
                        'type': 'one_hot',
                        'features': [
                            'contract_type', 'payment_method', 'subscription_type',
                            'online_security', 'tech_support', 'streaming_tv',
                            'streaming_movies', 'gender', 'partner', 'dependents'
                        ],
                        'scope': FeatureScope.CATEGORICAL
                    }
                ]
            )
        )
        
        # Add feature stage
        orchestrator.add_feature_stage(
            name="feature_engineering",
            config=feature_config,
            enabled=True
        )
        
        # Add training stage
        training_config = PipelineConfig(
            name="model_training_stage",
            description="Model training stage",
            model_config=ModelConfig(
                model_name="orchestrated_model",
                model_type="classification",
                hyperparameters={
                    'n_estimators': 50,
                    'max_depth': 5,
                    'min_samples_split': 5,
                    'random_state': 42
                },
                metrics=["accuracy", "precision", "recall", "f1"]
            )
        )
        
        orchestrator.add_training_stage(
            name="model_training",
            config=training_config,
            depends_on=["feature_engineering"],
            enabled=True
        )
        
        # In a real scenario, we would run the orchestrated pipeline
        # For testing, we'll just validate the orchestrator setup
        self.assertEqual(len(orchestrator.stages), 2)
        self.assertEqual(orchestrator.stages[0].name, "feature_engineering")
        self.assertEqual(orchestrator.stages[1].name, "model_training")
        self.assertEqual(len(orchestrator.stages[1].depends_on), 1)
        self.assertEqual(orchestrator.stages[1].depends_on[0], "feature_engineering")


if __name__ == "__main__":
    # Configure the unittest runner
    unittest.main()
