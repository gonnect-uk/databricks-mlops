#!/usr/bin/env python
"""
Customer Churn Prediction Model Example

This example demonstrates how to build a customer churn prediction model using
the Databricks MLOps framework with strong typing and Pydantic models.

The script showcases:
1. Type-safe configuration handling with Pydantic
2. Structured feature engineering pipeline
3. Strongly-typed model training and evaluation
4. MLflow integration with proper model tracking
5. Model deployment to a Databricks endpoint
"""
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import mlflow
import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from databricks_mlops.config import MLOpsConfigManager
from databricks_mlops.models.base import Result, StatusEnum
from databricks_mlops.models.config import ModelConfig
from databricks_mlops.pipelines import (FeatureEngineeringConfig, FeatureScope, FeatureTransformer,
                                       ModelDeployer, ModelDeploymentConfig, ModelFramework,
                                       ModelTrainer, ModelType, ScalerType, TrainingConfig)
from databricks_mlops.utils.logging import LogLevel, setup_logger
from databricks_mlops.workflows.mlflow_tracking import TrackingConfig


def load_sample_data() -> pd.DataFrame:
    """
    Load sample customer churn data.
    
    In a real scenario, this would load from Databricks tables or Delta Lake.
    Here we generate synthetic data for demonstration purposes.
    
    Returns:
        DataFrame with synthetic customer data
    """
    # Generate synthetic customer data
    np.random.seed(42)
    n_samples = 1000
    
    # Customer demographics
    tenure = np.random.randint(1, 73, n_samples)
    monthly_charges = np.random.uniform(20, 120, n_samples)
    total_charges = monthly_charges * (tenure + np.random.uniform(0, 0.2, n_samples) * tenure)
    
    contract_types = ['Month-to-month', 'One year', 'Two year']
    contract_type = np.random.choice(contract_types, n_samples, p=[0.6, 0.3, 0.1])
    
    payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
    payment_method = np.random.choice(payment_methods, n_samples)
    
    subscription_types = ['Basic', 'Standard', 'Premium']
    subscription_type = np.random.choice(subscription_types, n_samples)
    
    # Service features
    service_options = ['Yes', 'No']
    online_security = np.random.choice(service_options, n_samples, p=[0.4, 0.6])
    tech_support = np.random.choice(service_options, n_samples, p=[0.3, 0.7])
    streaming_tv = np.random.choice(service_options, n_samples, p=[0.5, 0.5])
    streaming_movies = np.random.choice(service_options, n_samples, p=[0.5, 0.5])
    
    # More demographics
    gender = np.random.choice(['Male', 'Female'], n_samples)
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    partner = np.random.choice(service_options, n_samples, p=[0.5, 0.5])
    dependents = np.random.choice(service_options, n_samples, p=[0.3, 0.7])
    
    # Target: churn (with dependencies on features)
    churn_prob = 0.2
    churn_prob += np.where(contract_type == 'Month-to-month', 0.2, 0)
    churn_prob -= np.where(tenure > 36, 0.15, 0)
    churn_prob += np.where(payment_method == 'Electronic check', 0.1, 0)
    churn_prob += np.where(monthly_charges > 80, 0.1, 0)
    churn_prob -= np.where(tech_support == 'Yes', 0.1, 0)
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    
    churned = (np.random.random(n_samples) < churn_prob).astype(int)
    
    # Create customer IDs
    customer_id = [f"CUST-{i:05d}" for i in range(1, n_samples + 1)]
    
    # Combine into DataFrame
    df = pd.DataFrame({
        'customer_id': customer_id,
        'tenure': tenure,
        'monthly_charges': monthly_charges, 
        'total_charges': total_charges,
        'contract_type': contract_type,
        'payment_method': payment_method,
        'subscription_type': subscription_type,
        'online_security': online_security,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'gender': gender,
        'senior_citizen': senior_citizen,
        'partner': partner,
        'dependents': dependents,
        'churned': churned,
        'last_update_time': datetime.now()
    })
    
    return df


def run_churn_prediction_example() -> None:
    """Run the full customer churn prediction example."""
    # Set up logging with proper configuration
    logger = setup_logger("churn_prediction", LogLevel.INFO)
    logger.info("Starting customer churn prediction example")
    
    # Load sample data
    logger.info("Loading customer data")
    data = load_sample_data()
    logger.info(f"Loaded {len(data)} customer records")
    
    # Split data for training
    X = data.drop(['customer_id', 'churned', 'last_update_time'], axis=1)
    y = data['churned']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create feature engineering configuration using Pydantic model
    logger.info("Setting up feature engineering pipeline")
    feature_config = FeatureEngineeringConfig(
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
                'name': 'binary_encoder',
                'type': 'one_hot',
                'features': [
                    'online_security', 'tech_support', 'streaming_tv',
                    'streaming_movies', 'gender', 'partner', 'dependents'
                ],
                'scope': FeatureScope.CATEGORICAL
            },
            {
                'name': 'categorical_encoder',
                'type': 'one_hot',
                'features': ['contract_type', 'payment_method', 'subscription_type'],
                'scope': FeatureScope.CATEGORICAL
            },
            {
                'name': 'service_combiner',
                'type': 'custom',
                'features': ['streaming_tv', 'streaming_movies'],
                'scope': FeatureScope.CUSTOM,
                'params': {
                    'output_feature': 'streaming_services',
                    'combine_method': 'count_yes'
                }
            }
        ]
    )
    
    # Initialize feature transformer with strong typing
    feature_transformer = FeatureTransformer(config=feature_config)
    
    # Transform features
    logger.info("Transforming features")
    X_train_transformed = feature_transformer.fit_transform(X_train)
    X_test_transformed = feature_transformer.transform(X_test)
    
    logger.info(f"Transformed training data shape: {X_train_transformed.shape}")
    
    # Create model training configuration using Pydantic model
    logger.info("Setting up model training pipeline")
    model_config = TrainingConfig(
        model_name="customer_churn_predictor",
        model_type=ModelType.CLASSIFICATION,
        framework=ModelFramework.SKLEARN,
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'bootstrap': True,
            'class_weight': 'balanced',
            'random_state': 42
        },
        metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
        validation_size=0.2,
        cv_folds=5,
        experiment_name="churn_prediction_example"
    )
    
    # Set up MLflow tracking
    tracking_config = TrackingConfig(
        tracking_uri=mlflow.get_tracking_uri(),
        experiment_name="churn_prediction_example",
        run_name=f"churn_model_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        tags={
            "model_type": "classification",
            "task": "churn_prediction",
            "framework": "sklearn"
        }
    )
    
    # Initialize model trainer with strong typing
    model_trainer = ModelTrainer(
        config=model_config,
        tracking_config=tracking_config
    )
    
    # Train the model with proper typing
    logger.info("Training churn prediction model")
    model = RandomForestClassifier(**model_config.hyperparameters)
    
    with mlflow.start_run(run_name=tracking_config.run_name) as run:
        # Train model
        model.fit(X_train_transformed, y_train)
        
        # Evaluate model with typed metrics
        y_pred = model.predict(X_test_transformed)
        y_prob = model.predict_proba(X_test_transformed)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        # Log typed metrics and model
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_train_transformed.columns.tolist(),
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")
        
        # Log model with proper signature
        mlflow.sklearn.log_model(
            model, 
            "model",
            input_example=X_train_transformed.iloc[0].to_dict(),
            registered_model_name="customer_churn_predictor"
        )
        
        logger.info(f"Model training completed. Metrics: {metrics}")
        logger.info(f"MLflow run ID: {run.info.run_id}")
        logger.info(f"MLflow artifact URI: {run.info.artifact_uri}")
    
    # Create deployment configuration with strong typing
    deployment_config = ModelDeploymentConfig(
        model_name="customer_churn_predictor",
        model_version="1",
        endpoint_name="customer-churn-endpoint",
        timeout_seconds=300
    )
    
    # Create model deployer
    model_deployer = ModelDeployer(config=deployment_config)
    
    # In a real scenario, this would deploy to Databricks
    logger.info("In a production environment, the model would be deployed to Databricks endpoint")
    logger.info(f"Deployment config: {deployment_config.json(indent=2)}")
    
    # Print feature importance for reference
    print("\nFeature Importance:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Print evaluation metrics
    print("\nModel Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    logger.info("Customer churn prediction example completed successfully")


if __name__ == "__main__":
    try:
        run_churn_prediction_example()
    except ValidationError as e:
        print(f"Validation error in configuration: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running example: {e}")
        sys.exit(1)
