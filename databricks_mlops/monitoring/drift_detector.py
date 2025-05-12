"""
Drift detection system with strong typing for monitoring data and model prediction changes.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from databricks_mlops.models.base import Result, StatusEnum, ValidationSeverity
from databricks_mlops.utils.logging import setup_logger

# Set up logger
logger = setup_logger("drift_detector")


class DriftType(str, Enum):
    """Types of drift that can be detected."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    FEATURE_DRIFT = "feature_drift"
    LABEL_DRIFT = "label_drift"


class DriftMethod(str, Enum):
    """Methods for detecting drift."""
    STATISTICAL_TEST = "statistical_test"
    DISTRIBUTION_DISTANCE = "distribution_distance"
    CLASSIFIER_MODEL = "classifier_model"
    THRESHOLD_BASED = "threshold_based"
    CUSTOM = "custom"


class DistanceMetric(str, Enum):
    """Distance metrics for measuring distribution differences."""
    KL_DIVERGENCE = "kl_divergence"
    JS_DIVERGENCE = "js_divergence"
    WASSERSTEIN = "wasserstein"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    POPULATION_STABILITY_INDEX = "population_stability_index"
    HELLINGER = "hellinger"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"


class StatisticalTest(str, Enum):
    """Statistical tests for drift detection."""
    T_TEST = "t_test"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    CHI_SQUARED = "chi_squared"
    ANDERSON_DARLING = "anderson_darling"
    MANN_WHITNEY = "mann_whitney"
    MOOD = "mood"
    WILCOXON = "wilcoxon"


class FeatureType(str, Enum):
    """Feature types for type-specific drift detection."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IMAGE = "image"
    TIME_SERIES = "time_series"
    GEOSPATIAL = "geospatial"
    MIXED = "mixed"


class DriftCheckConfig(BaseModel):
    """Configuration for a single drift check."""
    name: str
    description: Optional[str] = None
    feature_name: Optional[str] = None
    feature_type: FeatureType = FeatureType.NUMERICAL
    drift_type: DriftType
    method: DriftMethod
    distance_metric: Optional[DistanceMetric] = None
    statistical_test: Optional[StatisticalTest] = None
    threshold: float = 0.05
    severity: ValidationSeverity = ValidationSeverity.WARNING
    
    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: Optional[DistanceMetric], values: Dict[str, Any]) -> Optional[DistanceMetric]:
        """Validate distance metric is provided if required."""
        if values.get("method") == DriftMethod.DISTRIBUTION_DISTANCE and v is None:
            raise ValueError("distance_metric is required when method is distribution_distance")
        return v
    
    @field_validator("statistical_test")
    @classmethod
    def validate_statistical_test(cls, v: Optional[StatisticalTest], values: Dict[str, Any]) -> Optional[StatisticalTest]:
        """Validate statistical test is provided if required."""
        if values.get("method") == DriftMethod.STATISTICAL_TEST and v is None:
            raise ValueError("statistical_test is required when method is statistical_test")
        return v


class DriftDetectionConfig(BaseModel):
    """Configuration for drift detection."""
    reference_data_path: str
    current_data_path: str
    reference_window: Optional[str] = None  # e.g., "7d", "30d"
    current_window: Optional[str] = None
    drift_checks: List[DriftCheckConfig] = Field(default_factory=list)
    feature_mapping: Dict[str, FeatureType] = Field(default_factory=dict)
    sample_size: Optional[int] = None
    random_seed: int = 42
    
    @field_validator("drift_checks")
    @classmethod
    def validate_drift_checks(cls, v: List[DriftCheckConfig]) -> List[DriftCheckConfig]:
        """Validate at least one drift check is provided."""
        if not v:
            raise ValueError("At least one drift check must be provided")
        return v


class FeatureDriftResult(BaseModel):
    """Result of drift detection for a single feature."""
    feature_name: str
    feature_type: FeatureType
    drift_type: DriftType
    method: DriftMethod
    p_value: Optional[float] = None
    distance: Optional[float] = None
    threshold: float
    drift_detected: bool
    severity: ValidationSeverity
    reference_statistics: Dict[str, Any] = Field(default_factory=dict)
    current_statistics: Dict[str, Any] = Field(default_factory=dict)
    details: Dict[str, Any] = Field(default_factory=dict)


class DriftDetectionResult(Result):
    """Result of drift detection across features."""
    reference_data_path: str
    current_data_path: str
    detection_time: datetime = Field(default_factory=datetime.now)
    feature_results: List[FeatureDriftResult] = Field(default_factory=list)
    drift_detected: bool = False
    drift_score: float = 0.0
    highest_severity: ValidationSeverity = ValidationSeverity.INFO
    sample_size: Optional[int] = None


class DriftDetector:
    """
    Detects data and concept drift for model monitoring.
    
    This class provides methods to detect different types of drift
    between reference and current datasets.
    """
    
    def __init__(self, config: DriftDetectionConfig):
        """
        Initialize the drift detector.
        
        Args:
            config: Drift detection configuration
        """
        self.config = config
        self.logger = logger
    
    def detect_drift(self) -> DriftDetectionResult:
        """
        Detect drift according to the configuration.
        
        Returns:
            DriftDetectionResult: The result of drift detection
        """
        self.logger.info(f"Detecting drift between {self.config.reference_data_path} and {self.config.current_data_path}")
        
        # Initialize result
        result = DriftDetectionResult(
            status=StatusEnum.SUCCESS,
            message="Drift detection started",
            reference_data_path=self.config.reference_data_path,
            current_data_path=self.config.current_data_path
        )
        
        try:
            # Load reference and current data
            reference_data = self._load_data(self.config.reference_data_path)
            current_data = self._load_data(self.config.current_data_path)
            
            # Apply sampling if configured
            if self.config.sample_size:
                reference_data = self._sample_data(reference_data)
                current_data = self._sample_data(current_data)
            
            result.sample_size = len(reference_data) if self.config.sample_size else None
            
            # Perform each drift check
            feature_results = []
            for check in self.config.drift_checks:
                self.logger.info(f"Performing drift check: {check.name}")
                feature_result = self._perform_drift_check(check, reference_data, current_data)
                feature_results.append(feature_result)
                
                # Update overall drift status
                if feature_result.drift_detected and feature_result.severity.value > result.highest_severity.value:
                    result.highest_severity = feature_result.severity
            
            # Update the result
            result.feature_results = feature_results
            result.drift_detected = any(fr.drift_detected for fr in feature_results)
            result.drift_score = self._calculate_overall_drift_score(feature_results)
            
            # Set status based on drift severity
            if result.drift_detected:
                if result.highest_severity == ValidationSeverity.ERROR:
                    result.status = StatusEnum.FAILED
                elif result.highest_severity == ValidationSeverity.WARNING:
                    result.status = StatusEnum.WARNING
            
            # Update message
            if result.drift_detected:
                result.message = f"Drift detected with severity {result.highest_severity}"
            else:
                result.message = "No significant drift detected"
            
            self.logger.info(f"Drift detection completed: {result.message}")
            return result
            
        except Exception as e:
            error_msg = f"Error detecting drift: {str(e)}"
            self.logger.exception(error_msg)
            
            result.status = StatusEnum.FAILED
            result.message = error_msg
            result.errors = [{"type": type(e).__name__, "message": str(e)}]
            
            return result
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from the specified path.
        
        Args:
            data_path: Path to the data
            
        Returns:
            Loaded DataFrame
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Determine the file format
        # 2. Load the data using appropriate method
        
        self.logger.info(f"Loading data from {data_path}")
        
        # Placeholder: return a dummy DataFrame
        return pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "feature3": np.random.choice(["A", "B", "C"], 100)
        })
    
    def _sample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Sample data to the configured size.
        
        Args:
            data: DataFrame to sample
            
        Returns:
            Sampled DataFrame
        """
        if not self.config.sample_size or len(data) <= self.config.sample_size:
            return data
        
        return data.sample(self.config.sample_size, random_state=self.config.random_seed)
    
    def _perform_drift_check(
        self, check: DriftCheckConfig, reference_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> FeatureDriftResult:
        """
        Perform a single drift check.
        
        Args:
            check: Drift check configuration
            reference_data: Reference data
            current_data: Current data
            
        Returns:
            Result of the drift check
        """
        self.logger.info(f"Performing drift check for {check.feature_name or 'all features'}")
        
        # Extract feature data
        feature_name = check.feature_name or "all"
        
        # Get feature data
        if check.feature_name:
            if check.feature_name not in reference_data.columns:
                raise ValueError(f"Feature {check.feature_name} not found in reference data")
            if check.feature_name not in current_data.columns:
                raise ValueError(f"Feature {check.feature_name} not found in current data")
            
            reference_feature = reference_data[check.feature_name]
            current_feature = current_data[check.feature_name]
        else:
            # Use all features
            reference_feature = reference_data
            current_feature = current_data
        
        # Calculate statistics
        reference_stats = self._calculate_statistics(reference_feature, check.feature_type)
        current_stats = self._calculate_statistics(current_feature, check.feature_type)
        
        # Detect drift based on method
        if check.method == DriftMethod.STATISTICAL_TEST:
            p_value = self._compute_statistical_test(
                reference_feature, current_feature, check.statistical_test, check.feature_type
            )
            drift_detected = p_value < check.threshold
            return FeatureDriftResult(
                feature_name=feature_name,
                feature_type=check.feature_type,
                drift_type=check.drift_type,
                method=check.method,
                p_value=p_value,
                threshold=check.threshold,
                drift_detected=drift_detected,
                severity=check.severity,
                reference_statistics=reference_stats,
                current_statistics=current_stats,
                details={"test": check.statistical_test}
            )
            
        elif check.method == DriftMethod.DISTRIBUTION_DISTANCE:
            distance = self._compute_distribution_distance(
                reference_feature, current_feature, check.distance_metric, check.feature_type
            )
            drift_detected = distance > check.threshold
            return FeatureDriftResult(
                feature_name=feature_name,
                feature_type=check.feature_type,
                drift_type=check.drift_type,
                method=check.method,
                distance=distance,
                threshold=check.threshold,
                drift_detected=drift_detected,
                severity=check.severity,
                reference_statistics=reference_stats,
                current_statistics=current_stats,
                details={"metric": check.distance_metric}
            )
            
        elif check.method == DriftMethod.CLASSIFIER_MODEL:
            # This is a placeholder for classifier-based drift detection
            return FeatureDriftResult(
                feature_name=feature_name,
                feature_type=check.feature_type,
                drift_type=check.drift_type,
                method=check.method,
                threshold=check.threshold,
                drift_detected=False,  # Placeholder
                severity=check.severity,
                reference_statistics=reference_stats,
                current_statistics=current_stats,
                details={"method": "classifier_not_implemented"}
            )
            
        elif check.method == DriftMethod.THRESHOLD_BASED:
            # Compare simple statistics
            drift_detected = False
            details = {}
            
            # For numerical features, compare mean and std
            if check.feature_type == FeatureType.NUMERICAL:
                mean_diff = abs(reference_stats.get("mean", 0) - current_stats.get("mean", 0))
                std_diff = abs(reference_stats.get("std", 0) - current_stats.get("std", 0))
                drift_detected = mean_diff > check.threshold or std_diff > check.threshold
                details = {"mean_diff": mean_diff, "std_diff": std_diff}
            
            # For categorical features, compare value distribution
            elif check.feature_type == FeatureType.CATEGORICAL:
                max_diff = 0
                for category in set(reference_stats.get("value_counts", {}).keys()).union(
                    set(current_stats.get("value_counts", {}).keys())
                ):
                    ref_pct = reference_stats.get("value_counts", {}).get(category, 0)
                    cur_pct = current_stats.get("value_counts", {}).get(category, 0)
                    diff = abs(ref_pct - cur_pct)
                    max_diff = max(max_diff, diff)
                drift_detected = max_diff > check.threshold
                details = {"max_category_diff": max_diff}
            
            return FeatureDriftResult(
                feature_name=feature_name,
                feature_type=check.feature_type,
                drift_type=check.drift_type,
                method=check.method,
                threshold=check.threshold,
                drift_detected=drift_detected,
                severity=check.severity,
                reference_statistics=reference_stats,
                current_statistics=current_stats,
                details=details
            )
            
        else:
            # Unknown method
            self.logger.warning(f"Unknown drift detection method: {check.method}")
            return FeatureDriftResult(
                feature_name=feature_name,
                feature_type=check.feature_type,
                drift_type=check.drift_type,
                method=check.method,
                threshold=check.threshold,
                drift_detected=False,
                severity=check.severity,
                reference_statistics=reference_stats,
                current_statistics=current_stats,
                details={"error": f"Unknown method: {check.method}"}
            )
    
    def _calculate_statistics(self, data: Union[pd.DataFrame, pd.Series], feature_type: FeatureType) -> Dict[str, Any]:
        """
        Calculate statistics for a feature.
        
        Args:
            data: Feature data
            feature_type: Type of the feature
            
        Returns:
            Dictionary of statistics
        """
        if isinstance(data, pd.DataFrame):
            # For multi-feature data, calculate basic statistics for each column
            result = {}
            for col in data.columns:
                col_type = self.config.feature_mapping.get(col, feature_type)
                result[col] = self._calculate_statistics(data[col], col_type)
            return result
        
        # For single-feature data
        if feature_type == FeatureType.NUMERICAL:
            return {
                "mean": float(data.mean()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "median": float(data.median()),
                "q1": float(data.quantile(0.25)),
                "q3": float(data.quantile(0.75))
            }
        elif feature_type == FeatureType.CATEGORICAL:
            # Calculate value percentages
            value_counts = data.value_counts(normalize=True).to_dict()
            return {
                "unique_count": len(value_counts),
                "most_common": data.value_counts().index[0] if not data.empty else None,
                "most_common_pct": float(data.value_counts(normalize=True).iloc[0]) if not data.empty else 0,
                "value_counts": value_counts
            }
        elif feature_type == FeatureType.TEXT:
            # Basic text statistics
            if data.empty:
                return {"empty": True}
            
            # Convert to string if not already
            text_data = data.astype(str)
            return {
                "avg_length": float(text_data.str.len().mean()),
                "max_length": int(text_data.str.len().max()),
                "min_length": int(text_data.str.len().min()),
                "empty_count": int((text_data == "").sum()),
                "sample": text_data.iloc[0] if not text_data.empty else ""
            }
        else:
            # Other feature types not fully implemented
            return {"feature_type": feature_type}
    
    def _compute_statistical_test(
        self, reference: Union[pd.DataFrame, pd.Series], current: Union[pd.DataFrame, pd.Series],
        test: Optional[StatisticalTest], feature_type: FeatureType
    ) -> float:
        """
        Compute statistical test between distributions.
        
        Args:
            reference: Reference data
            current: Current data
            test: Statistical test to use
            feature_type: Type of the feature
            
        Returns:
            p-value of the test
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Use scipy or other statistical packages to perform the test
        # 2. Handle different feature types appropriately
        
        self.logger.info(f"Computing statistical test: {test}")
        
        # Example: return a dummy p-value
        return 0.03  # Example p-value
    
    def _compute_distribution_distance(
        self, reference: Union[pd.DataFrame, pd.Series], current: Union[pd.DataFrame, pd.Series],
        metric: Optional[DistanceMetric], feature_type: FeatureType
    ) -> float:
        """
        Compute distance between distributions.
        
        Args:
            reference: Reference data
            current: Current data
            metric: Distance metric to use
            feature_type: Type of the feature
            
        Returns:
            Distance between distributions
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Use scipy or other packages to compute distances
        # 2. Handle different feature types appropriately
        
        self.logger.info(f"Computing distribution distance: {metric}")
        
        # Example: return a dummy distance
        return 0.08  # Example distance
    
    def _calculate_overall_drift_score(self, feature_results: List[FeatureDriftResult]) -> float:
        """
        Calculate an overall drift score across all features.
        
        Args:
            feature_results: List of feature drift results
            
        Returns:
            Overall drift score
        """
        # This is a placeholder implementation
        # In a real implementation, you would:
        # 1. Combine individual feature drift scores
        # 2. Weight scores based on importance or severity
        
        # Simple implementation: average of drift indicators
        if not feature_results:
            return 0.0
        
        # Count drifted features with severity weights
        total_score = 0.0
        for result in feature_results:
            if result.drift_detected:
                if result.severity == ValidationSeverity.ERROR:
                    total_score += 1.0
                elif result.severity == ValidationSeverity.WARNING:
                    total_score += 0.5
                else:
                    total_score += 0.1
        
        # Normalize by number of features
        return total_score / len(feature_results)
