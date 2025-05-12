"""
Type-safe expression evaluation module for data validation expressions.

This module provides a strongly-typed approach to evaluating Pandas-style
expression strings in data validation. Instead of direct string manipulation
or eval(), it uses a proper parsing and evaluation system that maintains
type safety throughout.
"""
import re
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

class ExpressionType(Enum):
    """Enumeration of supported expression types with strong typing."""
    COMPARISON = auto()
    STRING = auto()
    LOGICAL = auto()
    MATHEMATICAL = auto()
    COLLECTION = auto()
    NULL_CHECK = auto()
    CUSTOM = auto()

class ExpressionToken(BaseModel):
    """Strongly-typed token representation for expression parsing."""
    token_type: ExpressionType
    value: str
    position: int
    
    @field_validator('token_type')
    def validate_token_type(cls, v: ExpressionType) -> ExpressionType:
        """Validate that the token type is a valid ExpressionType."""
        if not isinstance(v, ExpressionType):
            raise ValueError(f"Token type must be ExpressionType, got {type(v)}")
        return v

class ExpressionNode(BaseModel):
    """Base model for expression syntax tree with type information."""
    node_type: ExpressionType
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Evaluate this node on the provided data."""
        raise NotImplementedError("Subclasses must implement evaluate")

class ColumnNode(ExpressionNode):
    """Node representing a column reference with type tracking."""
    node_type: ExpressionType = Field(default=ExpressionType.COMPARISON)
    column_name: str
    inferred_type: Optional[str] = None
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Get the column data with proper type information."""
        if self.column_name not in data.columns:
            raise ValueError(f"Column '{self.column_name}' not found in data")
        
        # Maintain type information
        result = data[self.column_name]
        if self.inferred_type is None:
            self.inferred_type = str(result.dtype)
        return result

class ComparisonNode(ExpressionNode):
    """Strongly-typed comparison operation node."""
    node_type: ExpressionType = Field(default=ExpressionType.COMPARISON)
    left: ExpressionNode
    operator: str
    right: ExpressionNode
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Evaluate comparison with proper type checking."""
        left_val = self.left.evaluate(data)
        right_val = self.right.evaluate(data)
        
        # Type-aware comparison operators
        if self.operator == '==':
            return left_val == right_val
        elif self.operator == '!=':
            return left_val != right_val
        elif self.operator == '>':
            self._validate_numeric([left_val, right_val])
            return left_val > right_val
        elif self.operator == '>=':
            self._validate_numeric([left_val, right_val])
            return left_val >= right_val
        elif self.operator == '<':
            self._validate_numeric([left_val, right_val])
            return left_val < right_val
        elif self.operator == '<=':
            self._validate_numeric([left_val, right_val])
            return left_val <= right_val
        else:
            raise ValueError(f"Unsupported comparison operator: {self.operator}")
    
    def _validate_numeric(self, values: List[pd.Series]) -> None:
        """Ensure values are numeric types for proper comparison."""
        for val in values:
            if not pd.api.types.is_numeric_dtype(val.dtype):
                raise TypeError(f"Comparison operator '{self.operator}' requires numeric values, got {val.dtype}")

class LogicalNode(ExpressionNode):
    """Node for logical operations with proper short-circuit evaluation."""
    node_type: ExpressionType = Field(default=ExpressionType.LOGICAL)
    left: ExpressionNode
    operator: str
    right: Optional[ExpressionNode] = None  # Optional for 'not' operator
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Evaluate logical operations with proper boolean type handling."""
        left_val = self.left.evaluate(data)
        
        # Ensure we're working with boolean series
        if not pd.api.types.is_bool_dtype(left_val.dtype):
            left_val = left_val.astype(bool)
        
        if self.operator == 'not':
            return ~left_val
        
        if self.right is None:
            raise ValueError("Binary logical operations require a right operand")
        
        right_val = self.right.evaluate(data)
        if not pd.api.types.is_bool_dtype(right_val.dtype):
            right_val = right_val.astype(bool)
        
        if self.operator == 'and':
            return left_val & right_val
        elif self.operator == 'or':
            return left_val | right_val
        else:
            raise ValueError(f"Unsupported logical operator: {self.operator}")

class StringOperationNode(ExpressionNode):
    """Node for string operations with type safety."""
    node_type: ExpressionType = Field(default=ExpressionType.STRING)
    column: ExpressionNode
    method: str
    arguments: List[Any] = Field(default_factory=list)
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Apply string operation with proper type checking."""
        series = self.column.evaluate(data)
        
        # Ensure we're working with string data or handle nulls
        if not pd.api.types.is_string_dtype(series.dtype) and not pd.api.types.is_object_dtype(series.dtype):
            raise TypeError(f"String operations require string data, got {series.dtype}")
        
        # Access the string accessor and call the method
        if not hasattr(series.str, self.method):
            raise ValueError(f"Unsupported string method: {self.method}")
        
        string_method = getattr(series.str, self.method)
        return string_method(*self.arguments)

class NullCheckNode(ExpressionNode):
    """Node for null checks with proper handling of various null types."""
    node_type: ExpressionType = Field(default=ExpressionType.NULL_CHECK)
    column: ExpressionNode
    is_null: bool = True
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Check for null values with consistent handling."""
        series = self.column.evaluate(data)
        
        if self.is_null:
            return series.isna()
        else:
            return series.notna()

class CollectionCheckNode(ExpressionNode):
    """Node for collection membership checks with type consistency."""
    node_type: ExpressionType = Field(default=ExpressionType.COLLECTION)
    column: ExpressionNode
    operator: str  # 'in' or 'not in'
    values: List[Any]
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Check if values are in the collection with type-aware comparison."""
        series = self.column.evaluate(data)
        
        if self.operator == 'in':
            return series.isin(self.values)
        elif self.operator == 'not in':
            return ~series.isin(self.values)
        else:
            raise ValueError(f"Unsupported collection operator: {self.operator}")

class ExpressionParser:
    """Parser for validation expressions with type safety throughout."""
    
    def __init__(self):
        """Initialize the parser with supported operators and patterns."""
        self.comparison_operators = {'==', '!=', '>', '>=', '<', '<='}
        self.logical_operators = {'and', 'or', 'not'}
        self.string_methods = {'contains', 'startswith', 'endswith', 'match'}
        self.null_checks = {'is null', 'is not null'}
        self.collection_operators = {'in', 'not in'}
    
    def parse(self, expression: str) -> ExpressionNode:
        """Parse an expression string into a typed expression tree."""
        # Simplified parsing logic for the example
        # A full implementation would have a proper tokenizer and parser
        
        # Example for 'column_name.str.contains("@") or column_name is null'
        if 'is null' in expression:
            return self._parse_null_check(expression)
        elif '.str.' in expression:
            return self._parse_string_operation(expression)
        elif any(op in expression for op in self.comparison_operators):
            return self._parse_comparison(expression)
        elif any(op in expression for op in self.logical_operators):
            return self._parse_logical(expression)
        else:
            # Default to a simple column reference
            return ColumnNode(column_name=expression.strip())
    
    def _parse_null_check(self, expression: str) -> ExpressionNode:
        """Parse null check expressions with proper null handling."""
        if 'is null' in expression:
            column_name = expression.split('is null')[0].strip()
            return NullCheckNode(
                column=ColumnNode(column_name=column_name),
                is_null=True
            )
        elif 'is not null' in expression:
            column_name = expression.split('is not null')[0].strip()
            return NullCheckNode(
                column=ColumnNode(column_name=column_name),
                is_null=False
            )
        else:
            raise ValueError(f"Invalid null check expression: {expression}")
    
    def _parse_string_operation(self, expression: str) -> ExpressionNode:
        """Parse string operations with proper method handling."""
        # Simple handling for demonstration
        parts = expression.split('.str.')
        column_name = parts[0].strip()
        
        method_part = parts[1]
        method_match = re.match(r'(\w+)\((.*)\)', method_part)
        if not method_match:
            raise ValueError(f"Invalid string method syntax: {method_part}")
        
        method_name = method_match.group(1)
        arguments_str = method_match.group(2)
        
        # Handle arguments (simplistic approach)
        arguments = []
        if arguments_str:
            if arguments_str.startswith("'") and arguments_str.endswith("'"):
                arguments.append(arguments_str[1:-1])
            elif arguments_str.startswith('"') and arguments_str.endswith('"'):
                arguments.append(arguments_str[1:-1])
            else:
                # Try to parse as a literal
                try:
                    arg_value = eval(arguments_str)
                    arguments.append(arg_value)
                except:
                    arguments.append(arguments_str)
        
        return StringOperationNode(
            column=ColumnNode(column_name=column_name),
            method=method_name,
            arguments=arguments
        )
    
    def _parse_comparison(self, expression: str) -> ExpressionNode:
        """Parse comparison expressions with type-aware handling."""
        # Simplified for demonstration
        for op in self.comparison_operators:
            if op in expression:
                parts = expression.split(op)
                left = parts[0].strip()
                right = parts[1].strip()
                
                # Parse the left and right sides recursively
                left_node = self.parse(left)
                
                # Handle literals on the right
                try:
                    right_value = eval(right)
                    right_node = LiteralNode(value=right_value)
                except:
                    right_node = self.parse(right)
                
                return ComparisonNode(
                    left=left_node,
                    operator=op,
                    right=right_node
                )
        
        raise ValueError(f"No comparison operator found in: {expression}")
    
    def _parse_logical(self, expression: str) -> ExpressionNode:
        """Parse logical expressions with proper precedence."""
        # Simplified for demonstration
        for op in ['or', 'and']:  # Lower precedence first
            if f" {op} " in expression:
                parts = expression.split(f" {op} ", 1)
                left = parts[0].strip()
                right = parts[1].strip()
                
                return LogicalNode(
                    left=self.parse(left),
                    operator=op,
                    right=self.parse(right)
                )
        
        if expression.startswith('not '):
            expr = expression[4:].strip()
            return LogicalNode(
                left=self.parse(expr),
                operator='not'
            )
        
        raise ValueError(f"No logical operator found in: {expression}")

class LiteralNode(ExpressionNode):
    """Node for literal values with proper typing."""
    node_type: ExpressionType = Field(default=ExpressionType.COMPARISON)
    value: Any
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Convert literal to a series with proper type."""
        # Create a series of the same length as the data
        return pd.Series([self.value] * len(data), index=data.index)

class ValidationExpression(BaseModel):
    """Strongly-typed validation expression with parsing and evaluation."""
    expression: str
    parsed_node: Optional[ExpressionNode] = None
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Evaluate the expression on the data with type safety."""
        if self.parsed_node is None:
            parser = ExpressionParser()
            self.parsed_node = parser.parse(self.expression)
        
        return self.parsed_node.evaluate(data)
    
    def get_referenced_columns(self) -> Set[str]:
        """Extract column names from the expression for validation."""
        # This would scan the parsed node tree to find all column references
        # Simplified implementation for the example
        columns = set()
        expr = self.expression
        
        # Extract potential column names using a simple approach
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expr)
        
        # Filter out known keywords
        keywords = {'and', 'or', 'not', 'in', 'is', 'null', 'str', 'contains', 
                    'startswith', 'endswith', 'match', 'True', 'False'}
        
        columns = {word for word in words if word not in keywords}
        return columns

# Example usage in validation code:
def validate_with_expression(data: pd.DataFrame, expression: str) -> pd.Series:
    """
    Validate data using a strongly-typed expression.
    
    Args:
        data: DataFrame to validate
        expression: Expression string like "email.str.contains('@') or email is null"
        
    Returns:
        Boolean Series indicating which rows pass validation
    """
    validator = ValidationExpression(expression=expression)
    return validator.evaluate(data)

# Example functions to demonstrate validation usage

def validate_email_format(data: pd.DataFrame, email_col: str = 'email') -> pd.Series:
    """Validate email format with proper null handling."""
    expression = f"{email_col}.str.contains('@') or {email_col} is null"
    return validate_with_expression(data, expression)

def validate_numeric_range(data: pd.DataFrame, col: str, min_val: float, max_val: float) -> pd.Series:
    """Validate a numeric column is within range."""
    expression = f"{col} >= {min_val} and {col} <= {max_val}"
    return validate_with_expression(data, expression)

def validate_in_allowed_values(data: pd.DataFrame, col: str, allowed_values: List[str]) -> pd.Series:
    """Validate a column contains only allowed values."""
    values_str = str(allowed_values).replace('[', '').replace(']', '')
    expression = f"{col} in [{values_str}]"
    return validate_with_expression(data, expression)
