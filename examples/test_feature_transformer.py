#!/usr/bin/env python
"""
Unit Test Example for Feature Transformer Component

This example demonstrates how to properly test the feature transformer
component using pytest with a focus on strong typing and proper type validation.
It showcases how to test Pydantic models and transformations.
"""
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from databricks_mlops.pipelines import (EncoderType, FeatureEngineeringConfig,
                                      FeatureScope, FeatureTransformer,
                                      ImputerType, ScalerType, TransformerConfig,
                                      TransformerType)


class TestFeatureTransformer:
    """Test suite for the FeatureTransformer component with strong typing."""
    
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample data for testing with proper types."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 100
        
        # Create DataFrame with mixed types to test transformations
        data = pd.DataFrame({
            'id': [f"ID-{i}" for i in range(n_samples)],
            'numeric_no_missing': np.random.normal(0, 1, n_samples),
            'numeric_with_missing': np.random.normal(0, 1, n_samples),
            'categorical_low_cardinality': np.random.choice(['A', 'B', 'C'], n_samples),
            'categorical_high_cardinality': np.random.choice(
                [f"VAL-{i}" for i in range(20)], n_samples
            ),
            'binary_feature': np.random.choice([0, 1], n_samples),
            'constant_feature': np.ones(n_samples),
            'datetime_feature': pd.date_range(start='2020-01-01', periods=n_samples, freq='D'),
            'target': np.random.choice([0, 1], n_samples),
        })
        
        # Add missing values
        missing_idx = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
        data.loc[missing_idx, 'numeric_with_missing'] = np.nan
        
        return data

    @pytest.fixture
    def basic_config(self) -> FeatureEngineeringConfig:
        """Create a basic feature engineering configuration for testing."""
        return FeatureEngineeringConfig(
            categorical_features=[
                'categorical_low_cardinality',
                'categorical_high_cardinality'
            ],
            numerical_features=[
                'numeric_no_missing',
                'numeric_with_missing'
            ],
            binary_features=['binary_feature'],
            datetime_features=['datetime_feature'],
            id_features=['id'],
            target_column='target',
            transformers=[
                TransformerConfig(
                    name='numeric_scaler',
                    type=ScalerType.STANDARD,
                    features=['numeric_no_missing', 'numeric_with_missing'],
                    scope=FeatureScope.NUMERICAL
                ),
                TransformerConfig(
                    name='missing_imputer',
                    type=TransformerType.IMPUTER,
                    imputer_type=ImputerType.MEAN,
                    features=['numeric_with_missing'],
                    scope=FeatureScope.NUMERICAL
                ),
                TransformerConfig(
                    name='categorical_encoder',
                    type=TransformerType.ENCODER,
                    encoder_type=EncoderType.ONE_HOT,
                    features=['categorical_low_cardinality'],
                    scope=FeatureScope.CATEGORICAL,
                    handle_unknown='ignore'
                ),
                TransformerConfig(
                    name='high_card_encoder',
                    type=TransformerType.ENCODER,
                    encoder_type=EncoderType.TARGET,
                    features=['categorical_high_cardinality'],
                    scope=FeatureScope.CATEGORICAL,
                    target_column='target'
                ),
                TransformerConfig(
                    name='datetime_extractor',
                    type=TransformerType.DATETIME,
                    features=['datetime_feature'],
                    scope=FeatureScope.DATETIME,
                    extract_features=['year', 'month', 'day_of_week', 'is_weekend']
                )
            ]
        )

    def test_config_validation(self, basic_config: FeatureEngineeringConfig) -> None:
        """Test that configuration validation works correctly."""
        # Test valid configuration
        transformer = FeatureTransformer(config=basic_config)
        assert transformer.config is not None
        assert len(transformer.config.transformers) == 5
        
        # Test invalid configuration - Feature not in defined features
        invalid_config = basic_config.model_copy(deep=True)
        invalid_config.transformers[0].features = ['non_existent_feature']
        
        with pytest.raises(ValidationError):
            FeatureTransformer(config=invalid_config)
        
        # Test invalid configuration - Duplicate transformer names
        duplicate_config = basic_config.model_copy(deep=True)
        duplicate_config.transformers[1].name = duplicate_config.transformers[0].name
        
        with pytest.raises(ValueError, match="Duplicate transformer name"):
            FeatureTransformer(config=duplicate_config)

    def test_fit_transform(self, sample_data: pd.DataFrame, basic_config: FeatureEngineeringConfig) -> None:
        """Test the fit_transform method with strong typing validation."""
        # Initialize transformer
        transformer = FeatureTransformer(config=basic_config)
        
        # Fit and transform
        result_df = transformer.fit_transform(sample_data)
        
        # Validate result type
        assert isinstance(result_df, pd.DataFrame)
        
        # Validate that original ID column is preserved
        assert 'id' in result_df.columns
        
        # Validate that original target column is preserved
        assert 'target' in result_df.columns
        
        # Validate that numeric features are properly scaled
        assert 'numeric_no_missing' in result_df.columns
        assert result_df['numeric_no_missing'].mean() < 1e-10  # Close to zero mean
        assert abs(result_df['numeric_no_missing'].std() - 1.0) < 1e-10  # Close to unit variance
        
        # Validate that missing values are imputed
        assert 'numeric_with_missing' in result_df.columns
        assert not result_df['numeric_with_missing'].isna().any()
        
        # Validate one-hot encoding
        assert 'categorical_low_cardinality_A' in result_df.columns
        assert 'categorical_low_cardinality_B' in result_df.columns
        assert 'categorical_low_cardinality_C' in result_df.columns
        
        # Validate target encoding
        assert 'categorical_high_cardinality' in result_df.columns
        assert result_df['categorical_high_cardinality'].dtype == float
        
        # Validate datetime feature extraction
        assert 'datetime_feature_year' in result_df.columns
        assert 'datetime_feature_month' in result_df.columns
        assert 'datetime_feature_day_of_week' in result_df.columns
        assert 'datetime_feature_is_weekend' in result_df.columns

    def test_transform_only(self, sample_data: pd.DataFrame, basic_config: FeatureEngineeringConfig) -> None:
        """Test that transform without fit raises an error."""
        transformer = FeatureTransformer(config=basic_config)
        
        # Should raise error if transform is called before fit
        with pytest.raises(ValueError, match="Transformer has not been fitted"):
            transformer.transform(sample_data)
        
        # Now fit first
        transformer.fit(sample_data)
        
        # Then transform should work
        result_df = transformer.transform(sample_data)
        assert isinstance(result_df, pd.DataFrame)

    def test_feature_selection(self, sample_data: pd.DataFrame, basic_config: FeatureEngineeringConfig) -> None:
        """Test feature selection capabilities with proper typing."""
        # Add feature selection to config
        config_with_selection = basic_config.model_copy(deep=True)
        config_with_selection.transformers.append(
            TransformerConfig(
                name="variance_selector",
                type=TransformerType.FEATURE_SELECTOR,
                scope=FeatureScope.NUMERICAL,
                features=['numeric_no_missing', 'numeric_with_missing', 'constant_feature'],
                params={"threshold": 0.0}  # Removes features with 0 variance
            )
        )
        
        transformer = FeatureTransformer(config=config_with_selection)
        result_df = transformer.fit_transform(sample_data)
        
        # Constant feature should be removed
        assert 'constant_feature' not in result_df.columns
        
        # Non-constant features should still be present
        assert 'numeric_no_missing' in result_df.columns
        assert 'numeric_with_missing' in result_df.columns

    def test_serialization(self, sample_data: pd.DataFrame, basic_config: FeatureEngineeringConfig, tmp_path: Any) -> None:
        """Test serialization and deserialization of the transformer."""
        # Initialize and fit transformer
        transformer = FeatureTransformer(config=basic_config)
        transformer.fit(sample_data)
        
        # Save the transformer
        save_path = tmp_path / "transformer.pkl"
        transformer.save(str(save_path))
        
        # Load the transformer
        loaded_transformer = FeatureTransformer.load(str(save_path))
        
        # Transform with both transformers
        original_result = transformer.transform(sample_data)
        loaded_result = loaded_transformer.transform(sample_data)
        
        # Results should be identical
        pd.testing.assert_frame_equal(original_result, loaded_result)

    def test_get_feature_names(self, sample_data: pd.DataFrame, basic_config: FeatureEngineeringConfig) -> None:
        """Test getting output feature names with proper typing."""
        transformer = FeatureTransformer(config=basic_config)
        transformer.fit(sample_data)
        
        # Get feature names
        feature_names = transformer.get_feature_names()
        
        # Validate type
        assert isinstance(feature_names, List)
        assert all(isinstance(name, str) for name in feature_names)
        
        # Should include all transformed feature names
        assert 'numeric_no_missing' in feature_names
        assert 'numeric_with_missing' in feature_names
        assert 'categorical_low_cardinality_A' in feature_names
        assert 'datetime_feature_year' in feature_names

    def test_custom_transformer(self, sample_data: pd.DataFrame) -> None:
        """Test custom transformer implementation with strong typing."""
        # Create a config with a custom transformer
        config = FeatureEngineeringConfig(
            numerical_features=['numeric_no_missing', 'numeric_with_missing'],
            target_column='target',
            transformers=[
                TransformerConfig(
                    name='log_transformer',
                    type=TransformerType.CUSTOM,
                    features=['numeric_no_missing'],
                    scope=FeatureScope.NUMERICAL,
                    transform_function="lambda df, features: np.log1p(df[features])",
                    inverse_transform_function="lambda df, features: np.expm1(df[features])"
                )
            ]
        )
        
        # Ensure data is positive for log transform
        positive_data = sample_data.copy()
        positive_data['numeric_no_missing'] = np.abs(positive_data['numeric_no_missing']) + 0.1
        
        # Initialize transformer
        transformer = FeatureTransformer(config=config)
        
        # Fit and transform
        result_df = transformer.fit_transform(positive_data)
        
        # Validate log transform was applied
        original_values = positive_data['numeric_no_missing'].values
        transformed_values = result_df['numeric_no_missing'].values
        
        # Check log transform: log(x+1) should make values smaller but preserve order
        assert np.all(transformed_values < original_values)
        assert np.allclose(np.log1p(original_values), transformed_values)


if __name__ == "__main__":
    # Allow running as script for quick testing during development
    pytest.main(["-xvs", __file__])
