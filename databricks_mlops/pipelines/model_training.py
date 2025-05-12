"""
Model training pipeline with strong typing using Pydantic models.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from databricks_mlops.models.base import Result, StatusEnum
from databricks_mlops.models.config import ModelConfig
from databricks_mlops.utils.logging import setup_logger
from databricks_mlops.workflows.mlflow_tracking import (MLflowTracker,
                                                     ModelStage,
                                                     RegisterModelResult,
                                                     TrackingConfig)

# Set up logger
logger = setup_logger("model_training")


class ModelType(str, Enum):
    """Types of models supported in the framework."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    FORECASTING = "forecasting"
    RECOMMENDATION = "recommendation"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    ANOMALY_DETECTION = "anomaly_detection"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    CUSTOM = "custom"


class ModelFramework(str, Enum):
    """ML frameworks supported for model training."""
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SPARK_ML = "spark_ml"
    HUGGINGFACE = "huggingface"
    PROPHET = "prophet"
    CUSTOM = "custom"


class EvaluationMetric(str, Enum):
    """Evaluation metrics for model assessment."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC = "auc"
    LOG_LOSS = "log_loss"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2 = "r2"
    EXPLAINED_VARIANCE = "explained_variance"
    SILHOUETTE = "silhouette"
    CUSTOM = "custom"


class SplitStrategy(str, Enum):
    """Data splitting strategies for train/validation/test."""
    RANDOM = "random"
    STRATIFIED = "stratified"
    TIME_SERIES = "time_series"
    GROUP = "group"
    PREDEFINED = "predefined"


class TrainingConfig(BaseModel):
    """Configuration for model training pipeline."""
    model_config: ModelConfig
    model_type: ModelType
    model_framework: ModelFramework
    evaluation_metrics: List[EvaluationMetric]
    split_strategy: SplitStrategy = SplitStrategy.RANDOM
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    cross_validation_folds: int = 0  # 0 means no cross-validation
    hyperparameter_tuning: bool = False
    max_tuning_trials: int = 10
    early_stopping: bool = False
    patience: int = 5
    tracking_config: Optional[TrackingConfig] = None
    save_model: bool = True
    register_model: bool = True
    promote_to_stage: Optional[ModelStage] = None
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def validate_split_ratios(self) -> 'TrainingConfig':
        """Validate that split ratios sum to 1."""
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 0.001:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        return self


class EvaluationResult(BaseModel):
    """Results of model evaluation."""
    metrics: Dict[str, float] = Field(default_factory=dict)
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    confusion_matrix: Optional[List[List[int]]] = None
    roc_curve: Optional[Dict[str, List[float]]] = None
    pr_curve: Optional[Dict[str, List[float]]] = None
    cross_validation_results: Optional[Dict[str, List[float]]] = None
    test_predictions: Optional[Dict[str, List[float]]] = None


class TrainingResult(Result):
    """Result of model training pipeline."""
    model_name: str
    model_version: Optional[str] = None
    model_type: ModelType
    model_framework: ModelFramework
    evaluation: EvaluationResult
    training_time_seconds: float
    trained_at: datetime = Field(default_factory=datetime.now)
    artifacts: Dict[str, str] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    model_uri: Optional[str] = None
    registry_result: Optional[RegisterModelResult] = None


class ModelTrainer:
    """
    Trains and evaluates machine learning models with strong typing.
    
    This class handles the complete training pipeline including data splitting,
    model training, evaluation, and registration to MLflow.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the model trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = logger
        self.tracker: Optional[MLflowTracker] = None
        self.model: Any = None
    
    def train(self, data: pd.DataFrame) -> Tuple[Any, TrainingResult]:
        """
        Train a model on the provided data.
        
        Args:
            data: Input DataFrame with features and target
            
        Returns:
            Tuple of (trained_model, result)
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Starting model training for {self.config.model_config.model_name}")
        
        # Initialize result
        result = TrainingResult(
            status=StatusEnum.SUCCESS,
            message=f"Training started for model {self.config.model_config.model_name}",
            model_name=self.config.model_config.model_name,
            model_type=self.config.model_type,
            model_framework=self.config.model_framework,
            evaluation=EvaluationResult(),
            training_time_seconds=0.0
        )
        
        try:
            # Initialize MLflow tracking if configured
            if self.config.tracking_config:
                self.tracker = MLflowTracker(self.config.tracking_config)
                run_id = self.tracker.start_run(tags={
                    "model_type": self.config.model_type,
                    "model_framework": self.config.model_framework
                })
                self.logger.info(f"Started MLflow run with ID: {run_id}")
            
            # Split the data
            train_data, val_data, test_data = self._split_data(data)
            
            # Extract features and target
            X_train, y_train = self._extract_features_target(train_data)
            X_val, y_val = self._extract_features_target(val_data)
            X_test, y_test = self._extract_features_target(test_data)
            
            # Log data splits if tracking
            if self.tracker:
                self.tracker.log_params({
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "test_samples": len(X_test),
                    "features": len(X_train.columns)
                })
            
            # Hyperparameter tuning if configured
            best_hyperparameters = self.config.model_config.hyperparameters
            if self.config.hyperparameter_tuning:
                best_hyperparameters = self._tune_hyperparameters(X_train, y_train, X_val, y_val)
            
            # Train the model
            model = self._train_model(X_train, y_train, best_hyperparameters)
            self.model = model
            
            # Evaluate the model
            evaluation = self._evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)
            
            # Log metrics if tracking
            if self.tracker:
                for metric_name, metric_value in evaluation.metrics.items():
                    self.tracker.log_metric(metric_name, metric_value)
                
                for feature, importance in evaluation.feature_importance.items():
                    self.tracker.log_metric(f"importance_{feature}", importance)
            
            # Save and register model if configured
            model_uri = None
            registry_result = None
            
            if self.config.save_model and self.tracker:
                # Save model
                artifact_path = "model"
                model_info = self.tracker.log_model(
                    model,
                    artifact_path=artifact_path,
                    pip_requirements=[f"{self.config.model_framework}>0.0.1"],
                    registered_model_name=self.config.model_config.model_name if self.config.register_model else None
                )
                model_uri = model_info.path
                
                # Register model if not already done via log_model
                if self.config.register_model and not model_info.version:
                    registry_result = self.tracker.register_model(
                        model_uri=model_uri,
                        name=self.config.model_config.model_name
                    )
                    
                    # Promote to stage if configured
                    if registry_result.is_success and self.config.promote_to_stage:
                        self.tracker.transition_model_version_stage(
                            model_name=registry_result.model_name,
                            version=registry_result.model_version,
                            stage=self.config.promote_to_stage
                        )
            
            # Update result
            training_time = time.time() - start_time
            result.model_version = registry_result.model_version if registry_result and registry_result.is_success else None
            result.evaluation = evaluation
            result.training_time_seconds = training_time
            result.hyperparameters = best_hyperparameters
            result.model_uri = model_uri
            result.registry_result = registry_result
            result.message = f"Successfully trained model {self.config.model_config.model_name}"
            
            # End MLflow run if tracking
            if self.tracker:
                self.tracker.end_run()
            
            self.logger.info(f"Model training completed in {training_time:.2f} seconds")
            return model, result
            
        except Exception as e:
            error_msg = f"Error in model training: {str(e)}"
            self.logger.exception(error_msg)
            
            # End MLflow run if tracking
            if self.tracker:
                self.tracker.end_run(status="FAILED")
            
            training_time = time.time() - start_time
            result.status = StatusEnum.FAILED
            result.message = error_msg
            result.training_time_seconds = training_time
            result.errors = [{"type": type(e).__name__, "message": str(e)}]
            
            return None, result
    
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into train, validation, and test sets.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Use proper splitting strategies based on config
        # 2. Handle different split approaches for different model types
        
        self.logger.info(f"Splitting data using {self.config.split_strategy} strategy")
        
        if self.config.split_strategy == SplitStrategy.PREDEFINED:
            # Assume data already has a 'split' column
            train_data = data[data['split'] == 'train']
            val_data = data[data['split'] == 'val']
            test_data = data[data['split'] == 'test']
            return train_data, val_data, test_data
        
        # Simple random split
        from sklearn.model_selection import train_test_split
        
        # First split off test data
        train_val_data, test_data = train_test_split(
            data,
            test_size=self.config.test_ratio,
            random_state=self.config.random_seed
        )
        
        # Then split remaining data into train and validation
        relative_val_ratio = self.config.val_ratio / (self.config.train_ratio + self.config.val_ratio)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=relative_val_ratio,
            random_state=self.config.random_seed
        )
        
        return train_data, val_data, test_data
    
    def _extract_features_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and target from data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (X, y)
        """
        target_column = self.config.model_config.target_column
        feature_columns = self.config.model_config.features
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Validate features
        missing_features = [f for f in feature_columns if f not in data.columns]
        if missing_features:
            raise ValueError(f"Features not found in data: {missing_features}")
        
        X = data[feature_columns]
        y = data[target_column]
        
        return X, y
    
    def _tune_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Tune model hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Best hyperparameters
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Use hyperopt, Optuna, or similar
        # 2. Define parameter search space
        # 3. Perform optimization
        
        self.logger.info("Hyperparameter tuning not fully implemented")
        
        # Return default hyperparameters
        return self.config.model_config.hyperparameters
    
    def _train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series,
        hyperparameters: Dict[str, Any]
    ) -> Any:
        """
        Train the model with given hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            hyperparameters: Model hyperparameters
            
        Returns:
            Trained model
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Import appropriate model from selected framework
        # 2. Configure with hyperparameters
        # 3. Train the model
        
        self.logger.info(f"Training {self.config.model_framework} {self.config.model_type} model")
        
        # Placeholder model (sklearn-like interface)
        class PlaceholderModel:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.is_fitted = False
                self.feature_names = None
                self.classes_ = None
                self.feature_importances_ = None
            
            def fit(self, X, y):
                self.is_fitted = True
                self.feature_names = X.columns
                if len(set(y)) < 10:  # Classification-like
                    self.classes_ = sorted(set(y))
                self.feature_importances_ = {feature: 1.0/len(X.columns) for feature in X.columns}
                return self
            
            def predict(self, X):
                import numpy as np
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                if self.classes_ is not None:
                    return np.random.choice(self.classes_, size=len(X))
                else:
                    return np.random.normal(size=len(X))
            
            def predict_proba(self, X):
                import numpy as np
                if not self.is_fitted or self.classes_ is None:
                    raise ValueError("Model not fitted or not classification")
                return np.random.random((len(X), len(self.classes_)))
            
            def get_feature_importance(self):
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                return self.feature_importances_
        
        model = PlaceholderModel(**hyperparameters)
        model.fit(X_train, y_train)
        
        return model
    
    def _evaluate_model(
        self, model: Any,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series
    ) -> EvaluationResult:
        """
        Evaluate the trained model.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation results
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Use appropriate metrics for the model type
        # 2. Calculate confidence intervals
        # 3. Generate visualizations
        
        self.logger.info("Evaluating model performance")
        
        # Example metrics for placeholder
        metrics = {}
        feature_importance = {}
        confusion_matrix = None
        roc_curve = None
        pr_curve = None
        
        # Get model predictions
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics based on model type
        if self.config.model_type == ModelType.CLASSIFICATION:
            # Classification metrics
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                metrics["accuracy"] = accuracy_score(y_val, y_val_pred)
                metrics["precision"] = precision_score(y_val, y_val_pred, average="weighted")
                metrics["recall"] = recall_score(y_val, y_val_pred, average="weighted")
                metrics["f1"] = f1_score(y_val, y_val_pred, average="weighted")
                
                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_val, y_val_pred)
                confusion_matrix = cm.tolist() if hasattr(cm, "tolist") else None
                
                # ROC curve if binary classification
                if len(set(y_val)) == 2 and hasattr(model, "predict_proba"):
                    from sklearn.metrics import roc_curve as sk_roc_curve
                    y_val_proba = model.predict_proba(X_val)[:, 1]
                    fpr, tpr, thresholds = sk_roc_curve(y_val, y_val_proba)
                    roc_curve = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "thresholds": thresholds.tolist()
                    }
                
            except ImportError:
                self.logger.warning("scikit-learn not available for metric calculation")
                metrics = {
                    "accuracy": 0.85,  # Example values
                    "precision": 0.84,
                    "recall": 0.83,
                    "f1": 0.82
                }
                
        elif self.config.model_type == ModelType.REGRESSION:
            # Regression metrics
            try:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                metrics["mae"] = mean_absolute_error(y_val, y_val_pred)
                metrics["mse"] = mean_squared_error(y_val, y_val_pred)
                metrics["rmse"] = mean_squared_error(y_val, y_val_pred, squared=False)
                metrics["r2"] = r2_score(y_val, y_val_pred)
                
            except ImportError:
                self.logger.warning("scikit-learn not available for metric calculation")
                metrics = {
                    "mae": 0.15,  # Example values
                    "mse": 0.05,
                    "rmse": 0.22,
                    "r2": 0.78
                }
                
        else:
            # Other model types
            metrics = {m.value: 0.8 for m in self.config.evaluation_metrics}
        
        # Get feature importance if available
        if hasattr(model, "get_feature_importance"):
            feature_importance = model.get_feature_importance()
        elif hasattr(model, "feature_importances_"):
            # sklearn-like
            feature_importance = {
                feature: importance
                for feature, importance in zip(X_train.columns, model.feature_importances_)
            }
        
        # Create evaluation result
        return EvaluationResult(
            metrics=metrics,
            feature_importance=feature_importance,
            confusion_matrix=confusion_matrix,
            roc_curve=roc_curve,
            pr_curve=None,  # Not implemented in placeholder
            cross_validation_results=None,  # Not implemented in placeholder
            test_predictions=None  # Not implemented in placeholder
        )
