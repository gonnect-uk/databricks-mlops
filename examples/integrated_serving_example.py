#!/usr/bin/env python
"""
Integrated example showing complete MLOps workflow with model serving.

This example demonstrates how the type-safe model serving components 
integrate with the complete MLOps workflow, from data validation to
model training and deployment to serving.
"""
import os
import sys
import tempfile
from typing import Dict, List, Optional, Union

import mlflow
import pandas as pd
from pydantic import BaseModel, Field

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from databricks_mlops.models.config import DataConfig, ModelConfig, ValidationRule
from databricks_mlops.pipelines.data_validator import DataValidator
from databricks_mlops.pipelines.feature_engineer import FeatureTransformer
from databricks_mlops.utils.model_serving import (
    AuthType, EndpointConfig, EndpointCredentials, EndpointType,
    TabularModelClient, create_model_client
)
from databricks_mlops.utils.logging import setup_logger

class IntegratedMLOpsExample:
    """
    Example class demonstrating the complete MLOps workflow with model serving.
    
    This shows how type safety is maintained throughout the entire process,
    from data validation to model deployment and serving.
    """
    
    def __init__(self):
        """Initialize the example with logger."""
        self.logger = setup_logger("integrated_mlops")
        # Set tracking URI for local development (would be set to Databricks in production)
        mlflow.set_tracking_uri("file:" + os.path.join(tempfile.gettempdir(), "mlruns"))
    
    def run_complete_workflow(self, 
                             data_path: str,
                             workspace_url: Optional[str] = None, 
                             token: Optional[str] = None) -> None:
        """
        Run a complete MLOps workflow with model serving.
        
        Args:
            data_path: Path to input data CSV
            workspace_url: Optional Databricks workspace URL
            token: Optional authentication token
        """
        self.logger.info("Starting integrated MLOps workflow with model serving")
        
        # STEP 1: Data Validation with strongly-typed validation rules
        validation_rules = [
            ValidationRule(
                name="no_missing_values",
                condition="feature1 is not null and feature2 is not null",
                severity="error",
                description="Critical features should not have missing values"
            ),
            ValidationRule(
                name="valid_feature_range",
                condition="feature1 >= 0 and feature1 <= 100",
                severity="warning",
                description="Feature1 should be between 0 and 100"
            )
        ]
        
        data_config = DataConfig(
            source_path=data_path,
            destination_path=os.path.join(tempfile.gettempdir(), "validated_data.csv"),
            validation_rules=validation_rules
        )
        
        # Run validation with type safety
        validator = DataValidator(data_config)
        validation_result = validator.validate()
        
        if validation_result.success:
            self.logger.info("Data validation passed")
            validated_data = validation_result.data
        else:
            self.logger.error(f"Data validation failed: {validation_result.error_message}")
            return
        
        # STEP 2: Feature Engineering with type safety
        feature_config = {
            "numeric_features": ["feature1", "feature2"],
            "categorical_features": ["category"],
            "target": "target"
        }
        
        transformer = FeatureTransformer(feature_config)
        
        # Fit and transform with proper type handling
        transformer.fit(validated_data)
        transformed_data = transformer.transform(validated_data)
        
        # STEP 3: Model Training with MLflow tracking
        model_config = ModelConfig(
            model_type="classification",
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 5
            },
            metrics=["accuracy", "f1_score", "precision", "recall"]
        )
        
        # Start an MLflow run with proper typing
        with mlflow.start_run(run_name="integrated-example") as run:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            # Prepare data with proper type handling
            X = transformed_data.drop("target", axis=1)
            y = transformed_data["target"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=model_config.hyperparameters["n_estimators"],
                max_depth=model_config.hyperparameters["max_depth"]
            )
            model.fit(X_train, y_train)
            
            # Log model and metrics with proper typing
            mlflow.sklearn.log_model(model, "model")
            
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            y_pred = model.predict(X_test)
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred, average="weighted"),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted")
            }
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            model_uri = f"runs:/{run.info.run_id}/model"
            
            self.logger.info(f"Model trained and logged to: {model_uri}")
            self.logger.info(f"Model metrics: {metrics}")
        
        # STEP 4: Model Serving (mock or real depending on provided credentials)
        if workspace_url and token:
            # Configure endpoint with type safety
            endpoint_config = EndpointConfig(
                endpoint_name="integrated-example-endpoint",
                endpoint_type=EndpointType.SERVING,
                model_name="integrated-example",
                model_version="1",
                scale_to_zero_enabled=True,
                min_instances=1,
                max_instances=1
            )
            
            # Create credentials with type safety
            credentials = EndpointCredentials(
                auth_type=AuthType.TOKEN,
                token=token
            )
            
            # Create client
            client = TabularModelClient(
                workspace_url=workspace_url,
                credentials=credentials,
                logger=self.logger
            )
            
            # Create test data for prediction
            test_data = pd.DataFrame({
                "feature1": [10, 20, 30],
                "feature2": [0.5, 1.5, 2.5],
                "category": ["A", "B", "C"]
            })
            
            # Transform test data
            test_features = transformer.transform(test_data)
            
            try:
                # Make predictions with type safety
                self.logger.info("Making predictions using model serving endpoint")
                predictions = client.predict(
                    endpoint_name=endpoint_config.endpoint_name,
                    features=test_features.drop("target", axis=1, errors="ignore")
                )
                
                self.logger.info(f"Predictions: {predictions}")
                
            except Exception as e:
                self.logger.error(f"Error making predictions: {str(e)}")
        else:
            # Run with mock predictions for demonstration
            self.logger.info("Performing mock prediction (no Databricks credentials provided)")
            
            test_data = pd.DataFrame({
                "feature1": [10, 20, 30],
                "feature2": [0.5, 1.5, 2.5],
                "category": ["A", "B", "C"]
            })
            
            # Transform test data
            test_features = transformer.transform(test_data)
            
            # Use local model for prediction
            predictions = model.predict(test_features.drop("target", axis=1, errors="ignore"))
            
            self.logger.info(f"Mock predictions: {predictions}")
    
    def run_with_sample_data(self) -> None:
        """Run the workflow with sample data when real data isn't available."""
        self.logger.info("Creating sample data for demonstration")
        
        # Create temporary sample data
        sample_data = pd.DataFrame({
            "feature1": [10, 20, 30, 40, 50],
            "feature2": [0.5, 1.5, 2.5, 3.5, 4.5],
            "category": ["A", "B", "A", "C", "B"],
            "target": [0, 1, 0, 1, 1]
        })
        
        sample_path = os.path.join(tempfile.gettempdir(), "sample_data.csv")
        sample_data.to_csv(sample_path, index=False)
        
        self.run_complete_workflow(sample_path)


def main() -> None:
    """Run the integrated MLOps example with model serving."""
    example = IntegratedMLOpsExample()
    
    # Check if real credentials are provided as environment variables
    workspace_url = os.environ.get("DATABRICKS_WORKSPACE_URL")
    token = os.environ.get("DATABRICKS_TOKEN")
    data_path = os.environ.get("DATA_PATH")
    
    if data_path:
        # Run with real data
        print("Running with provided data path")
        example.run_complete_workflow(data_path, workspace_url, token)
    else:
        # Run with sample data
        print("Running with generated sample data")
        example.run_with_sample_data()


if __name__ == "__main__":
    main()
