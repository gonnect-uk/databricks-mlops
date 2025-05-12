#!/usr/bin/env python
"""
Example demonstrating the strongly-typed validation expression language.

This example shows how the framework evaluates validation expressions like
"email.str.contains('@') or email is null" in a type-safe manner.
"""
import os
import sys
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from databricks_mlops.utils.expression_evaluator import ValidationExpression, validate_with_expression


class ValidationRule(BaseModel):
    """Strongly-typed validation rule."""
    name: str
    condition: str
    severity: str = "error"
    description: Optional[str] = None


def print_validation_results(data: pd.DataFrame, rule: ValidationRule) -> None:
    """Print validation results in a tabular format."""
    validator = ValidationExpression(expression=rule.condition)
    results = validator.evaluate(data)
    
    # Combine data with validation results
    result_df = data.copy()
    result_df['validation_passed'] = results
    
    print(f"\n=== Validation Rule: {rule.name} ===")
    print(f"Condition: {rule.condition}")
    if rule.description:
        print(f"Description: {rule.description}")
    print(f"Severity: {rule.severity}")
    print("\nResults:")
    
    # Highlight failed rows
    passed_rows = result_df[result_df['validation_passed']]
    failed_rows = result_df[~result_df['validation_passed']]
    
    print(f"✓ {len(passed_rows)} rows passed validation")
    print(f"✗ {len(failed_rows)} rows failed validation")
    
    if not failed_rows.empty:
        print("\nFailed rows:")
        print(failed_rows.drop('validation_passed', axis=1))


def main() -> None:
    """Run the expression language validation example."""
    # Sample data with different types and validation scenarios
    data = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005', None],
        'email': ['user@example.com', 'invalid-email', None, 'another@test.com', 'missing-at.com', 'third@example.com'],
        'age': [25, 16, 42, -5, 30, 22],
        'tenure': [12, 0, -1, 24, 6, 3],
        'monthly_charges': [50.0, 30.0, 70.0, 65.0, 45.0, 55.0],
        'total_charges': [600.0, 30.0, 69.0, 1560.0, 270.0, 165.0],
        'status': ['active', 'pending', 'invalid', 'active', 'pending', 'active']
    })
    
    # Define validation rules using our expression language
    rules = [
        ValidationRule(
            name="no_missing_ids",
            condition="customer_id is not null",
            severity="error",
            description="Customer ID should never be null"
        ),
        ValidationRule(
            name="valid_email",
            condition="email.str.contains('@') or email is null",
            severity="warning",
            description="Email should be valid format or null"
        ),
        ValidationRule(
            name="positive_age",
            condition="age > 0",
            severity="error",
            description="Age must be positive"
        ),
        ValidationRule(
            name="non_negative_tenure",
            condition="tenure >= 0",
            severity="error",
            description="Tenure cannot be negative"
        ),
        ValidationRule(
            name="charge_consistency",
            condition="total_charges >= monthly_charges or tenure == 0",
            severity="error",
            description="Total charges should be at least equal to monthly charges (unless tenure is 0)"
        ),
        ValidationRule(
            name="valid_status",
            condition="status in ['active', 'pending', 'closed']",
            severity="error",
            description="Status must be one of the allowed values"
        ),
        ValidationRule(
            name="adult_or_pending",
            condition="(age >= 18) or (status == 'pending')",
            severity="warning",
            description="Users under 18 should have pending status"
        )
    ]
    
    # Print the sample data
    print("Sample Data:")
    print(data)
    print("\nRunning validation with type-safe expression language...")
    
    # Run and print validation for each rule
    for rule in rules:
        print_validation_results(data, rule)
    
    print("\nComplex validation example:")
    complex_rule = ValidationRule(
        name="complex_business_rule",
        condition="(age >= 18 and status == 'active') or (age < 18 and status == 'pending' and email is not null)",
        severity="error",
        description="Adults can be active, minors must be pending with email"
    )
    print_validation_results(data, complex_rule)


if __name__ == "__main__":
    main()
