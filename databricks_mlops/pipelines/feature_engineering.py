"""
Feature engineering pipeline with strong typing using Pydantic models.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from databricks_mlops.models.base import Result, StatusEnum
from databricks_mlops.models.config import FeatureConfig
from databricks_mlops.utils.logging import setup_logger

# Set up logger
logger = setup_logger("feature_engineering")


class TransformerType(str, Enum):
    """Types of feature transformers."""
    SCALER = "scaler"
    ENCODER = "encoder"
    IMPUTER = "imputer"
    OUTLIER_DETECTOR = "outlier_detector"
    DIMENSION_REDUCER = "dimension_reducer"
    TEXT_VECTORIZER = "text_vectorizer"
    CUSTOM = "custom"


class ScalerType(str, Enum):
    """Types of scalers for numerical features."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    MAXABS = "maxabs"
    NORMALIZER = "normalizer"
    QUANTILE = "quantile"
    LOG = "log"
    POWER = "power"


class EncoderType(str, Enum):
    """Types of encoders for categorical features."""
    ONE_HOT = "one_hot"
    ORDINAL = "ordinal"
    TARGET = "target"
    BINARY = "binary"
    FREQUENCY = "frequency"
    HASH = "hash"
    EMBEDDING = "embedding"


class ImputerType(str, Enum):
    """Types of imputers for missing values."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    KNN = "knn"
    ITERATIVE = "iterative"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"


class OutlierDetectorType(str, Enum):
    """Types of outlier detectors."""
    IQR = "iqr"
    Z_SCORE = "z_score"
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    DBSCAN = "dbscan"


class DimensionReducerType(str, Enum):
    """Types of dimension reducers."""
    PCA = "pca"
    KERNEL_PCA = "kernel_pca"
    TRUNCATED_SVD = "truncated_svd"
    LDA = "lda"
    TSNE = "tsne"
    UMAP = "umap"


class TextVectorizerType(str, Enum):
    """Types of text vectorizers."""
    COUNT = "count"
    TF_IDF = "tf_idf"
    WORD2VEC = "word2vec"
    GLOVE = "glove"
    BERT = "bert"
    FASTTEXT = "fasttext"
    UNIVERSAL_SENTENCE_ENCODER = "universal_sentence_encoder"


class FeatureScope(str, Enum):
    """Scope of feature transformation."""
    ALL = "all"
    SELECTED = "selected"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATE = "date"
    MISSING = "missing"
    OUTLIER = "outlier"


class TransformerConfig(BaseModel):
    """Configuration for a feature transformer."""
    name: str
    description: Optional[str] = None
    transformer_type: TransformerType
    specific_type: Optional[Union[
        ScalerType, EncoderType, ImputerType, 
        OutlierDetectorType, DimensionReducerType, 
        TextVectorizerType, str
    ]] = None
    scope: FeatureScope = FeatureScope.ALL
    features: Optional[List[str]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    input_check: bool = True
    output_check: bool = True
    handle_errors: bool = True
    
    @field_validator("features")
    @classmethod
    def validate_features(cls, v: Optional[List[str]], values: Dict[str, Any]) -> Optional[List[str]]:
        """Validate that features list is provided if scope is 'selected'."""
        if values.get("scope") == FeatureScope.SELECTED and (not v or len(v) == 0):
            raise ValueError("features must be provided when scope is 'selected'")
        return v
    
    @field_validator("specific_type")
    @classmethod
    def validate_specific_type(cls, v: Optional[Any], values: Dict[str, Any]) -> Optional[Any]:
        """Validate that specific_type matches transformer_type."""
        transformer_type = values.get("transformer_type")
        if transformer_type == TransformerType.CUSTOM:
            # Custom transformers can have any specific_type
            return v
            
        if transformer_type == TransformerType.SCALER and v is not None:
            if not isinstance(v, ScalerType):
                if isinstance(v, str) and v in ScalerType.__members__:
                    return ScalerType(v)
                raise ValueError(f"specific_type {v} is not a valid ScalerType")
                
        elif transformer_type == TransformerType.ENCODER and v is not None:
            if not isinstance(v, EncoderType):
                if isinstance(v, str) and v in EncoderType.__members__:
                    return EncoderType(v)
                raise ValueError(f"specific_type {v} is not a valid EncoderType")
                
        elif transformer_type == TransformerType.IMPUTER and v is not None:
            if not isinstance(v, ImputerType):
                if isinstance(v, str) and v in ImputerType.__members__:
                    return ImputerType(v)
                raise ValueError(f"specific_type {v} is not a valid ImputerType")
                
        elif transformer_type == TransformerType.OUTLIER_DETECTOR and v is not None:
            if not isinstance(v, OutlierDetectorType):
                if isinstance(v, str) and v in OutlierDetectorType.__members__:
                    return OutlierDetectorType(v)
                raise ValueError(f"specific_type {v} is not a valid OutlierDetectorType")
                
        elif transformer_type == TransformerType.DIMENSION_REDUCER and v is not None:
            if not isinstance(v, DimensionReducerType):
                if isinstance(v, str) and v in DimensionReducerType.__members__:
                    return DimensionReducerType(v)
                raise ValueError(f"specific_type {v} is not a valid DimensionReducerType")
                
        elif transformer_type == TransformerType.TEXT_VECTORIZER and v is not None:
            if not isinstance(v, TextVectorizerType):
                if isinstance(v, str) and v in TextVectorizerType.__members__:
                    return TextVectorizerType(v)
                raise ValueError(f"specific_type {v} is not a valid TextVectorizerType")
                
        return v


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering pipeline."""
    feature_config: FeatureConfig
    transformers: List[TransformerConfig] = Field(default_factory=list)
    add_missing_indicator: bool = False
    handle_unknown_categories: bool = True
    pass_through_features: List[str] = Field(default_factory=list)
    output_format: str = "delta"
    output_mode: str = "overwrite"
    register_feature_store: bool = True
    save_transformers: bool = True
    transformers_path: Optional[str] = None


class FeatureTransformationResult(Result):
    """Result of feature transformation."""
    input_shape: Tuple[int, int]
    output_shape: Tuple[int, int]
    feature_mapping: Dict[str, List[str]] = Field(default_factory=dict)
    created_features: List[str] = Field(default_factory=list)
    dropped_features: List[str] = Field(default_factory=list)
    execution_time_seconds: float
    transformation_time: datetime = Field(default_factory=datetime.now)
    transformers_saved: bool = False
    transformers_path: Optional[str] = None
    feature_store_registered: bool = False
    feature_table_name: Optional[str] = None


class FeatureTransformer:
    """
    Transforms features according to the configured transformations.
    
    This class handles the application of feature transformers to input data,
    with strong typing and error handling.
    """
    
    def __init__(self, config: FeatureEngineeringConfig):
        """
        Initialize the feature transformer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
        self.logger = logger
        self.transformers: Dict[str, Any] = {}
    
    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, FeatureTransformationResult]:
        """
        Apply transformations to the input data.
        
        Args:
            data: Input DataFrame to transform
            
        Returns:
            Tuple of (transformed_data, result)
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Transforming features for {len(data)} rows")
        
        # Initialize result
        result = FeatureTransformationResult(
            status=StatusEnum.SUCCESS,
            message="Feature transformation started",
            input_shape=(len(data), len(data.columns)),
            output_shape=(0, 0),
            execution_time_seconds=0.0
        )
        
        try:
            # Initialize tracking variables
            transformed_data = data.copy()
            created_features = []
            dropped_features = []
            feature_mapping = {}
            
            # Apply each transformer in sequence
            for transformer_config in self.config.transformers:
                self.logger.info(f"Applying transformer: {transformer_config.name}")
                
                # Get features to transform
                features_to_transform = self._get_features_to_transform(transformer_config, transformed_data)
                
                if not features_to_transform:
                    self.logger.warning(f"No features selected for transformer {transformer_config.name}, skipping")
                    continue
                
                # Apply the transformer
                try:
                    input_cols = features_to_transform
                    transformer_result = self._apply_transformer(
                        transformer_config, transformed_data, features_to_transform
                    )
                    
                    if transformer_result is None:
                        self.logger.warning(f"Transformer {transformer_config.name} returned None, skipping")
                        continue
                    
                    transformed_data, output_cols = transformer_result
                    
                    # Track created and dropped features
                    new_features = [col for col in output_cols if col not in input_cols]
                    dropped = [col for col in input_cols if col not in output_cols]
                    
                    created_features.extend(new_features)
                    dropped_features.extend(dropped)
                    
                    # Track feature mapping
                    for input_col in input_cols:
                        related_output_cols = [col for col in output_cols if input_col in col]
                        feature_mapping[input_col] = related_output_cols
                    
                except Exception as e:
                    error_msg = f"Error applying transformer {transformer_config.name}: {str(e)}"
                    self.logger.error(error_msg)
                    
                    if transformer_config.handle_errors:
                        self.logger.warning(f"Continuing despite error in transformer {transformer_config.name}")
                    else:
                        raise RuntimeError(error_msg) from e
            
            # Add pass-through features
            for feature in self.config.pass_through_features:
                if feature in data.columns and feature not in transformed_data.columns:
                    transformed_data[feature] = data[feature]
                    feature_mapping[feature] = [feature]
            
            # Save transformers if configured
            transformers_saved = False
            transformers_path = None
            if self.config.save_transformers and self.config.transformers_path:
                transformers_saved, transformers_path = self._save_transformers()
            
            # Register to feature store if configured
            feature_store_registered = False
            if self.config.register_feature_store:
                feature_store_registered = self._register_feature_store(transformed_data)
            
            # Update result
            execution_time = time.time() - start_time
            result.output_shape = (len(transformed_data), len(transformed_data.columns))
            result.feature_mapping = feature_mapping
            result.created_features = created_features
            result.dropped_features = dropped_features
            result.execution_time_seconds = execution_time
            result.transformers_saved = transformers_saved
            result.transformers_path = transformers_path
            result.feature_store_registered = feature_store_registered
            result.feature_table_name = self.config.feature_config.feature_table_name
            result.message = f"Successfully transformed features from {result.input_shape[1]} to {result.output_shape[1]} columns"
            
            self.logger.info(f"Feature transformation completed in {execution_time:.2f} seconds")
            return transformed_data, result
            
        except Exception as e:
            error_msg = f"Error in feature transformation: {str(e)}"
            self.logger.exception(error_msg)
            
            execution_time = time.time() - start_time
            result.status = StatusEnum.FAILED
            result.message = error_msg
            result.execution_time_seconds = execution_time
            result.errors = [{"type": type(e).__name__, "message": str(e)}]
            
            return data, result
    
    def _get_features_to_transform(self, config: TransformerConfig, data: pd.DataFrame) -> List[str]:
        """
        Get the list of features to transform based on the transformer scope.
        
        Args:
            config: Transformer configuration
            data: Input DataFrame
            
        Returns:
            List of column names to transform
        """
        if config.scope == FeatureScope.SELECTED and config.features:
            # Use explicitly selected features
            return [f for f in config.features if f in data.columns]
            
        elif config.scope == FeatureScope.ALL:
            # Use all columns
            return list(data.columns)
            
        elif config.scope == FeatureScope.NUMERICAL:
            # Use numerical columns
            return list(data.select_dtypes(include=["number"]).columns)
            
        elif config.scope == FeatureScope.CATEGORICAL:
            # Use categorical columns
            return list(data.select_dtypes(include=["object", "category"]).columns)
            
        elif config.scope == FeatureScope.TEXT:
            # Use object columns (as a proxy for text)
            return list(data.select_dtypes(include=["object"]).columns)
            
        elif config.scope == FeatureScope.DATE:
            # Use datetime columns
            return list(data.select_dtypes(include=["datetime"]).columns)
            
        elif config.scope == FeatureScope.MISSING:
            # Use columns with missing values
            return [col for col in data.columns if data[col].isna().any()]
            
        elif config.scope == FeatureScope.OUTLIER:
            # For outlier scope, use numerical columns by default
            return list(data.select_dtypes(include=["number"]).columns)
            
        else:
            self.logger.warning(f"Unknown feature scope: {config.scope}")
            return []
    
    def _apply_transformer(
        self, config: TransformerConfig, data: pd.DataFrame, features: List[str]
    ) -> Optional[Tuple[pd.DataFrame, List[str]]]:
        """
        Apply a transformer to the data.
        
        Args:
            config: Transformer configuration
            data: Input DataFrame
            features: Features to transform
            
        Returns:
            Tuple of (transformed_data, output_features) or None if error
        """
        try:
            # Create transformer if needed
            if config.name not in self.transformers:
                self.transformers[config.name] = self._create_transformer(config)
            
            transformer = self.transformers[config.name]
            
            # For placeholders only - in a real implementation, these would be actual transformers
            if transformer is None:
                return None
            
            # Example placeholder transformation - in a real implementation this would use the actual transformer
            result_df = data.copy()
            output_features = []
            
            # Different transformations based on transformer type
            if config.transformer_type == TransformerType.SCALER:
                # For numerical scaling, we keep same column names
                for feature in features:
                    if feature in result_df.columns:
                        # Placeholder scaling operation
                        result_df[feature] = (result_df[feature] - result_df[feature].mean()) / result_df[feature].std()
                        output_features.append(feature)
                
            elif config.transformer_type == TransformerType.ENCODER:
                # For categorical encoding, we might create new columns
                for feature in features:
                    if feature in result_df.columns:
                        # Placeholder encoding operation (one-hot-like)
                        unique_values = result_df[feature].dropna().unique()
                        for value in unique_values:
                            new_col = f"{feature}_{value}"
                            result_df[new_col] = (result_df[feature] == value).astype(int)
                            output_features.append(new_col)
                        
                        # Drop original column for one-hot encoding
                        if config.specific_type == EncoderType.ONE_HOT:
                            result_df = result_df.drop(columns=[feature])
                        else:
                            output_features.append(feature)
                
            elif config.transformer_type == TransformerType.IMPUTER:
                # For imputation, we keep same column names
                for feature in features:
                    if feature in result_df.columns:
                        # Placeholder imputation operation
                        result_df[feature] = result_df[feature].fillna(result_df[feature].mean() if result_df[feature].dtype.kind in 'fc' else result_df[feature].mode()[0])
                        output_features.append(feature)
                
            elif config.transformer_type == TransformerType.OUTLIER_DETECTOR:
                # For outlier detection, we might create indicator columns
                for feature in features:
                    if feature in result_df.columns:
                        # Placeholder outlier operation
                        if result_df[feature].dtype.kind in 'fc':  # Numerical
                            mean = result_df[feature].mean()
                            std = result_df[feature].std()
                            lower_bound = mean - 3 * std
                            upper_bound = mean + 3 * std
                            
                            # Create outlier indicator
                            result_df[f"{feature}_outlier"] = ((result_df[feature] < lower_bound) | (result_df[feature] > upper_bound)).astype(int)
                            output_features.append(feature)
                            output_features.append(f"{feature}_outlier")
                        else:
                            output_features.append(feature)
                
            elif config.transformer_type == TransformerType.DIMENSION_REDUCER:
                # For dimension reduction, we create new columns
                # This is a very simplified placeholder
                if len(features) > 0:
                    # Create a few "principal components"
                    for i in range(min(3, len(features))):
                        result_df[f"PC_{i+1}"] = result_df[features].mean(axis=1) + i * 0.1
                        output_features.append(f"PC_{i+1}")
                
            elif config.transformer_type == TransformerType.TEXT_VECTORIZER:
                # For text vectorization, we create new columns
                for feature in features:
                    if feature in result_df.columns:
                        # Placeholder vectorization
                        result_df[f"{feature}_length"] = result_df[feature].astype(str).str.len()
                        result_df[f"{feature}_word_count"] = result_df[feature].astype(str).str.split().str.len()
                        
                        output_features.append(f"{feature}_length")
                        output_features.append(f"{feature}_word_count")
                
            elif config.transformer_type == TransformerType.CUSTOM:
                # For custom, we just pass through the features
                output_features = features
                
            else:
                self.logger.warning(f"Unknown transformer type: {config.transformer_type}")
                return None
            
            return result_df, output_features
            
        except Exception as e:
            error_msg = f"Error applying transformer {config.name}: {str(e)}"
            self.logger.error(error_msg)
            
            if config.handle_errors:
                # Return original data and features
                return data, features
            else:
                raise RuntimeError(error_msg) from e
    
    def _create_transformer(self, config: TransformerConfig) -> Any:
        """
        Create a transformer based on configuration.
        
        Args:
            config: Transformer configuration
            
        Returns:
            Transformer object or None if not implemented
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Import appropriate libraries (scikit-learn, etc.)
        # 2. Create the transformer with configured parameters
        # 3. Return the transformer
        
        self.logger.info(f"Creating transformer {config.name} of type {config.transformer_type}")
        
        # Return a placeholder
        return {"name": config.name, "type": config.transformer_type, "config": config}
    
    def _save_transformers(self) -> Tuple[bool, Optional[str]]:
        """
        Save transformers to disk.
        
        Returns:
            Tuple of (success, path)
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Use pickle, joblib, or other serialization
        # 2. Save transformers to the configured path
        
        if not self.config.transformers_path:
            self.logger.warning("No transformers_path provided for saving transformers")
            return False, None
        
        try:
            path = self.config.transformers_path
            self.logger.info(f"Saving transformers to {path}")
            
            # Placeholder for saving logic
            return True, path
            
        except Exception as e:
            self.logger.error(f"Error saving transformers: {str(e)}")
            return False, None
    
    def _register_feature_store(self, data: pd.DataFrame) -> bool:
        """
        Register features to the feature store.
        
        Args:
            data: Transformed data to register
            
        Returns:
            Success status
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Connect to the Databricks Feature Store
        # 2. Register the features
        
        try:
            table_name = self.config.feature_config.feature_table_name
            self.logger.info(f"Registering features to feature store: {table_name}")
            
            # Placeholder for registration logic
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering features to feature store: {str(e)}")
            return False
