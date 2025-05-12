"""
Data validation utilities with strong typing using Pydantic models.
"""
import logging
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from databricks_mlops.models.base import StatusEnum, ValidationResult, ValidationSeverity
from databricks_mlops.models.config import DataValidationRule
from databricks_mlops.utils.logging import setup_logger

# Set up logger
logger = setup_logger("data_validation")


class DataValidationError(Exception):
    """Exception raised for errors in data validation."""
    pass


class ValidationMethod(str, Enum):
    """Validation methods supported by the framework."""
    PANDAS_QUERY = "pandas_query"
    GREAT_EXPECTATIONS = "great_expectations"
    SQL_CHECK = "sql_check"
    CUSTOM_PYTHON = "custom_python"


class ValidationScope(str, Enum):
    """Scope of the validation (what portion of data to validate)."""
    FULL = "full"
    SAMPLE = "sample"
    RECENT = "recent"
    INCREMENTAL = "incremental"


class ValidationCheck(BaseModel):
    """Strongly-typed model for a data validation check."""
    name: str
    description: Optional[str] = None
    condition: str
    method: ValidationMethod = ValidationMethod.PANDAS_QUERY
    severity: ValidationSeverity = ValidationSeverity.ERROR
    scope: ValidationScope = ValidationScope.FULL
    sample_size: Optional[int] = None
    sample_fraction: Optional[float] = None
    
    @field_validator("sample_fraction")
    @classmethod
    def validate_sample_fraction(cls, v: Optional[float]) -> Optional[float]:
        """Validate that sample_fraction is between 0 and 1."""
        if v is not None and (v <= 0 or v > 1):
            raise ValueError("sample_fraction must be between 0 and 1")
        return v
    
    @field_validator("sample_size", "sample_fraction")
    @classmethod
    def validate_sampling_parameters(cls, v: Optional[Union[int, float]], values: Dict[str, Any]) -> Optional[Union[int, float]]:
        """Validate that sampling parameters are set appropriately for the scope."""
        if values.get("scope") == ValidationScope.SAMPLE and v is None and "sample_size" not in values and "sample_fraction" not in values:
            raise ValueError("Either sample_size or sample_fraction must be provided when scope is 'sample'")
        return v


class ValidationOptions(BaseModel):
    """Options for data validation."""
    fail_on_error: bool = True
    fail_on_warning: bool = False
    throw_exception: bool = False
    log_results: bool = True
    log_level_failure: str = "ERROR"
    log_level_success: str = "INFO"
    log_data_on_failure: bool = False
    max_log_data_rows: int = 10


class ValidationResult(ValidationResult):  # Extends base ValidationResult
    """Enhanced validation result with data-specific fields."""
    dataset_name: Optional[str] = None
    validation_time_ms: Optional[float] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    sample_count: Optional[int] = None
    failure_examples: List[Dict[str, Any]] = Field(default_factory=list)


class DataValidator:
    """
    Validates data quality using strongly-typed validation rules.
    
    This class provides methods to validate data against a set of rules
    using different validation methods.
    """
    
    def __init__(self, options: Optional[ValidationOptions] = None):
        """
        Initialize the data validator.
        
        Args:
            options: Options for validation behavior
        """
        self.options = options or ValidationOptions()
        self.logger = logger
    
    def validate_with_rules(
        self,
        data: Union[pd.DataFrame, str],  # DataFrame or SQL table reference
        validation_rules: List[Union[ValidationCheck, DataValidationRule, str]],
        dataset_name: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate data against a list of validation rules.
        
        Args:
            data: The data to validate (DataFrame or SQL table reference)
            validation_rules: List of validation rules to check
            dataset_name: Optional name of the dataset for logging
            
        Returns:
            ValidationResult: The result of the validation
            
        Raises:
            DataValidationError: If validation fails and throw_exception is True
        """
        import time
        start_time = time.time()
        
        # Track validation results
        passed_validations = []
        failed_validations = []
        all_rules = []
        has_error = False
        has_warning = False
        
        # Process the validation rules
        normalized_rules = self._normalize_rules(validation_rules)
        
        try:
            # Get DataFrame if input is a SQL reference
            df = self._get_dataframe(data)
            
            # Execute each validation rule
            for rule in normalized_rules:
                all_rules.append(rule.name)
                validation_result = self._execute_validation_check(df, rule)
                
                if validation_result["passed"]:
                    passed_validations.append(rule.name)
                else:
                    failure_info = {
                        "rule": rule.name,
                        "condition": rule.condition,
                        "severity": rule.severity,
                        "message": validation_result.get("message", "Validation failed")
                    }
                    
                    # Add failure examples if available and configured
                    if self.options.log_data_on_failure and "failure_examples" in validation_result:
                        failure_info["examples"] = validation_result["failure_examples"]
                    
                    failed_validations.append(failure_info)
                    
                    # Track severity levels
                    if rule.severity == ValidationSeverity.ERROR:
                        has_error = True
                    elif rule.severity == ValidationSeverity.WARNING:
                        has_warning = True
            
            # Determine validation status
            status = StatusEnum.SUCCESS
            if has_error and self.options.fail_on_error:
                status = StatusEnum.FAILED
            elif has_warning and self.options.fail_on_warning:
                status = StatusEnum.WARNING
            elif has_warning:
                status = StatusEnum.WARNING
            
            # Create the validation result
            validation_time_ms = (time.time() - start_time) * 1000
            result = ValidationResult(
                status=status,
                message=self._create_result_message(status, len(passed_validations), len(failed_validations)),
                validation_type="data_quality",
                validation_rules=all_rules,
                passed_validations=passed_validations,
                failed_validations=failed_validations,
                severity=ValidationSeverity.ERROR if has_error else ValidationSeverity.WARNING if has_warning else ValidationSeverity.INFO,
                dataset_name=dataset_name,
                validation_time_ms=validation_time_ms,
                row_count=len(df),
                column_count=len(df.columns),
            )
            
            # Log the validation result
            if self.options.log_results:
                log_level = self.options.log_level_failure if status == StatusEnum.FAILED else self.options.log_level_success
                log_message = f"Data validation for {dataset_name or 'dataset'}: {result.message}"
                
                if log_level.upper() == "ERROR":
                    self.logger.error(log_message)
                elif log_level.upper() == "WARNING":
                    self.logger.warning(log_message)
                else:
                    self.logger.info(log_message)
            
            # Raise exception if configured and validation failed
            if self.options.throw_exception and status == StatusEnum.FAILED:
                raise DataValidationError(result.message)
            
            return result
            
        except DataValidationError:
            # Re-raise DataValidationError
            raise
        except Exception as e:
            error_msg = f"Error during data validation: {str(e)}"
            self.logger.exception(error_msg)
            
            if self.options.throw_exception:
                raise DataValidationError(error_msg) from e
            
            # Create failure result
            return ValidationResult(
                status=StatusEnum.FAILED,
                message=error_msg,
                validation_type="data_quality",
                validation_rules=all_rules,
                passed_validations=passed_validations,
                failed_validations=[{
                    "rule": "validation_execution",
                    "message": error_msg
                }],
                severity=ValidationSeverity.ERROR,
                dataset_name=dataset_name
            )
    
    def _normalize_rules(
        self, rules: List[Union[ValidationCheck, DataValidationRule, str]]
    ) -> List[ValidationCheck]:
        """
        Normalize different rule formats to ValidationCheck objects.
        
        Args:
            rules: List of rules in different formats
            
        Returns:
            List of normalized ValidationCheck objects
        """
        normalized = []
        
        for i, rule in enumerate(rules):
            if isinstance(rule, ValidationCheck):
                normalized.append(rule)
            elif isinstance(rule, DataValidationRule):
                # Convert DataValidationRule to ValidationCheck
                normalized.append(ValidationCheck(
                    name=rule.name,
                    description=rule.description,
                    condition=rule.condition,
                    method=ValidationMethod.PANDAS_QUERY,
                    severity=rule.severity,
                    scope=ValidationScope.FULL
                ))
            elif isinstance(rule, str):
                # Create a ValidationCheck from a string condition
                normalized.append(ValidationCheck(
                    name=f"rule_{i+1}",
                    condition=rule,
                    method=ValidationMethod.PANDAS_QUERY,
                    severity=ValidationSeverity.ERROR,
                    scope=ValidationScope.FULL
                ))
            else:
                raise ValueError(f"Unsupported validation rule type: {type(rule)}")
        
        return normalized
    
    def _get_dataframe(self, data: Union[pd.DataFrame, str]) -> pd.DataFrame:
        """
        Get a DataFrame from the input data.
        
        Args:
            data: DataFrame or SQL table reference
            
        Returns:
            DataFrame to validate
        """
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, str):
            # Assume it's a SQL table reference and try to read it
            # This is a placeholder - in a real implementation, you'd use
            # Databricks SQL or other methods to read the table
            raise NotImplementedError("SQL table reference not implemented")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _execute_validation_check(self, df: pd.DataFrame, rule: ValidationCheck) -> Dict[str, Any]:
        """
        Execute a validation check against a DataFrame.
        
        Args:
            df: DataFrame to validate
            rule: Validation rule to check
            
        Returns:
            Dictionary with validation results
        """
        # Apply sampling if specified
        validation_df = self._apply_sampling(df, rule)
        
        try:
            if rule.method == ValidationMethod.PANDAS_QUERY:
                # Use pandas query to validate
                valid_rows = validation_df.query(rule.condition)
                passed = len(valid_rows) == len(validation_df)
                
                result = {
                    "passed": passed,
                    "message": f"Validation {rule.name} {'passed' if passed else 'failed'}"
                }
                
                # If validation failed and logging is enabled, capture examples
                if not passed and self.options.log_data_on_failure:
                    invalid_rows = validation_df[~validation_df.index.isin(valid_rows.index)]
                    result["failure_examples"] = invalid_rows.head(
                        self.options.max_log_data_rows
                    ).to_dict(orient="records")
                
                return result
                
            elif rule.method == ValidationMethod.GREAT_EXPECTATIONS:
                # Placeholder for Great Expectations integration
                self.logger.warning("Great Expectations validation not implemented yet")
                return {"passed": False, "message": "Method not implemented"}
                
            elif rule.method == ValidationMethod.SQL_CHECK:
                # Placeholder for SQL check integration
                self.logger.warning("SQL check validation not implemented yet")
                return {"passed": False, "message": "Method not implemented"}
                
            elif rule.method == ValidationMethod.CUSTOM_PYTHON:
                # Placeholder for custom Python function validation
                self.logger.warning("Custom Python validation not implemented yet")
                return {"passed": False, "message": "Method not implemented"}
                
            else:
                return {"passed": False, "message": f"Unsupported validation method: {rule.method}"}
                
        except Exception as e:
            return {
                "passed": False,
                "message": f"Error executing validation rule {rule.name}: {str(e)}"
            }
    
    def _apply_sampling(self, df: pd.DataFrame, rule: ValidationCheck) -> pd.DataFrame:
        """
        Apply sampling to the DataFrame based on the rule scope.
        
        Args:
            df: Original DataFrame
            rule: Validation rule with sampling configuration
            
        Returns:
            Sampled DataFrame for validation
        """
        if rule.scope == ValidationScope.FULL:
            # Use the full dataset
            return df
            
        elif rule.scope == ValidationScope.SAMPLE:
            # Sample the dataset
            if rule.sample_size is not None:
                # Sample a specific number of rows
                sample_size = min(rule.sample_size, len(df))
                return df.sample(n=sample_size)
                
            elif rule.sample_fraction is not None:
                # Sample a fraction of rows
                return df.sample(frac=rule.sample_fraction)
                
            else:
                # Default to 10% sample
                return df.sample(frac=0.1)
                
        elif rule.scope == ValidationScope.RECENT:
            # Assume DataFrame has a timestamp column and sort by it
            # This is a placeholder - in a real implementation, you'd identify
            # the timestamp column and sort appropriately
            self.logger.warning("Recent data sampling not implemented yet")
            return df
            
        elif rule.scope == ValidationScope.INCREMENTAL:
            # Validate only new/changed data
            # This is a placeholder - in a real implementation, you'd determine
            # what data is new/changed since the last validation
            self.logger.warning("Incremental validation not implemented yet")
            return df
            
        else:
            # Default to full dataset
            return df
    
    def _create_result_message(self, status: StatusEnum, passed_count: int, failed_count: int) -> str:
        """
        Create a human-readable message for the validation result.
        
        Args:
            status: The validation status
            passed_count: Number of passed validations
            failed_count: Number of failed validations
            
        Returns:
            Human-readable message
        """
        total_count = passed_count + failed_count
        
        if status == StatusEnum.SUCCESS:
            return f"All {total_count} validation rules passed"
        elif status == StatusEnum.WARNING:
            return f"{passed_count} of {total_count} validation rules passed, {failed_count} warnings"
        else:
            return f"{passed_count} of {total_count} validation rules passed, {failed_count} errors"
