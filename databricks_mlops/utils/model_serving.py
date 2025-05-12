"""
Strongly-typed utilities for interacting with Databricks model serving endpoints.

This module provides Pydantic-based, type-safe wrappers around the Databricks
serving endpoint API. It ensures that all interactions with endpoints maintain
proper type information and validation throughout the process.
"""
import json
import logging
import time
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, Field, field_validator, validator
from pydantic.generics import GenericModel

from databricks_mlops.utils.logging import setup_logger

# Type definitions for strong typing
T = TypeVar('T')
PredictionType = TypeVar('PredictionType')
InputType = TypeVar('InputType')

class EndpointType(str, Enum):
    """Strongly-typed enum for endpoint types."""
    SERVING = "serving"
    SERVERLESS = "serverless"
    STANDARD = "standard"
    
class EndpointStatus(str, Enum):
    """Strongly-typed enum for endpoint status."""
    CREATING = "CREATING"
    READY = "READY"
    UPDATING = "UPDATING"
    FAILED = "FAILED"
    
class EndpointScaleType(str, Enum):
    """Strongly-typed enum for endpoint scaling types."""
    FIXED = "FIXED"
    AUTO = "AUTO"

class AuthType(str, Enum):
    """Strongly-typed enum for authentication types."""
    TOKEN = "token"
    AAD = "aad"
    SERVICE_PRINCIPAL = "service_principal"
    
class ModelInputSchema(BaseModel):
    """Base model for endpoint input schema with type information."""
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('input_schema')
    def validate_schema(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the schema is properly structured."""
        # Ensure schema has required fields
        if 'columns' not in v and 'inputs' not in v:
            raise ValueError("Schema must contain 'columns' or 'inputs' field")
        return v

class ModelOutputSchema(BaseModel):
    """Base model for endpoint output schema with type information."""
    output_schema: Dict[str, Any] = Field(default_factory=dict)

class EndpointConfig(BaseModel):
    """Strongly-typed configuration for Databricks model serving endpoints."""
    endpoint_name: str
    endpoint_type: EndpointType = EndpointType.SERVING
    model_name: str
    model_version: Union[str, int]
    workspaceid: Optional[str] = None
    scale_to_zero_enabled: bool = True
    min_instances: int = 1
    max_instances: int = 1
    timeout_seconds: int = 300
    
    @field_validator('min_instances')
    def validate_min_instances(cls, v: int) -> int:
        """Validate that min_instances is within acceptable range."""
        if v < 0:
            raise ValueError("min_instances must be non-negative")
        return v
    
    @field_validator('max_instances')
    def validate_max_instances(cls, v: int, values: Dict[str, Any]) -> int:
        """Validate that max_instances is greater than or equal to min_instances."""
        if 'min_instances' in values and v < values['min_instances']:
            raise ValueError("max_instances must be >= min_instances")
        return v

class EndpointCredentials(BaseModel):
    """Strongly-typed credentials for authenticating with endpoints."""
    auth_type: AuthType = AuthType.TOKEN
    token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = None
    
    @field_validator('token')
    def validate_token(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate that token is provided for token auth."""
        if values.get('auth_type') == AuthType.TOKEN and not v:
            raise ValueError("Token must be provided for token authentication")
        return v
    
    @field_validator('client_id', 'client_secret', 'tenant_id')
    def validate_service_principal(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate that service principal details are provided for SP auth."""
        if values.get('auth_type') == AuthType.SERVICE_PRINCIPAL and not v:
            raise ValueError("client_id, client_secret, and tenant_id must be provided for service principal auth")
        return v

class PredictionRequest(GenericModel, Generic[InputType]):
    """Strongly-typed generic model for prediction requests."""
    inputs: InputType
    params: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary in the format expected by Databricks API."""
        if isinstance(self.inputs, pd.DataFrame):
            # Convert DataFrame to the format expected by Databricks
            return {
                "dataframe_records": self.inputs.to_dict(orient="records"),
                "params": self.params or {}
            }
        elif isinstance(self.inputs, Dict):
            # Handle dictionary inputs (for tensor-based models)
            return {
                "inputs": self.inputs,
                "params": self.params or {}
            }
        elif isinstance(self.inputs, List):
            # Handle list inputs (for arrays)
            return {
                "inputs": self.inputs,
                "params": self.params or {}
            }
        else:
            raise TypeError(f"Unsupported input type: {type(self.inputs)}")

class PredictionResponse(GenericModel, Generic[PredictionType]):
    """Strongly-typed generic model for prediction responses."""
    predictions: PredictionType
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionResponse[PredictionType]':
        """Create PredictionResponse from dictionary with type safety."""
        predictions = data.get('predictions', None)
        metadata = data.get('metadata', {})
        
        if predictions is None:
            raise ValueError("Response does not contain 'predictions'")
        
        return cls(predictions=predictions, metadata=metadata)

class ModelServingClient:
    """Strongly-typed client for interacting with Databricks model serving endpoints."""
    
    def __init__(
        self, 
        workspace_url: str,
        credentials: EndpointCredentials,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the model serving client with endpoint details.
        
        Args:
            workspace_url: Databricks workspace URL
            credentials: Authentication credentials
            logger: Optional logger instance
        """
        self.workspace_url = workspace_url.rstrip('/')
        self.credentials = credentials
        self.logger = logger or setup_logger("model_serving")
        self._token: Optional[str] = None
        
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers with proper token handling."""
        if self.credentials.auth_type == AuthType.TOKEN:
            if not self.credentials.token:
                raise ValueError("Token not provided for token authentication")
            return {"Authorization": f"Bearer {self.credentials.token}"}
        elif self.credentials.auth_type == AuthType.SERVICE_PRINCIPAL:
            # In a real implementation, this would get a token using MSAL
            # For demonstration, we'll just raise an error
            raise NotImplementedError("Service principal auth not implemented in this example")
        else:
            raise ValueError(f"Unsupported auth type: {self.credentials.auth_type}")
    
    def get_endpoint_status(self, endpoint_name: str) -> EndpointStatus:
        """
        Get the current status of a model serving endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            
        Returns:
            EndpointStatus enum value
        """
        url = f"{self.workspace_url}/api/2.0/serving-endpoints/{endpoint_name}"
        headers = self._get_auth_headers()
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            self.logger.error(f"Failed to get endpoint status: {response.text}")
            raise ValueError(f"Failed to get endpoint status: {response.text}")
        
        data = response.json()
        state = data.get('state', {}).get('ready')
        
        if state:
            return EndpointStatus.READY
        elif data.get('state', {}).get('failed'):
            return EndpointStatus.FAILED
        elif data.get('state', {}).get('creating'):
            return EndpointStatus.CREATING
        else:
            return EndpointStatus.UPDATING
    
    def predict_pandas(
        self, 
        endpoint_name: str, 
        dataframe: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Make a prediction using a DataFrame as input.
        
        Args:
            endpoint_name: Name of the endpoint
            dataframe: Pandas DataFrame with features
            params: Optional parameters for the model
            
        Returns:
            DataFrame with prediction results
        """
        request = PredictionRequest[Dict[str, List[Dict[str, Any]]]](
            inputs={"dataframe_records": dataframe.to_dict(orient="records")},
            params=params
        )
        
        return self._make_prediction(endpoint_name, request)
    
    def predict_dict(
        self, 
        endpoint_name: str, 
        data_dict: Dict[str, Any], 
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Make a prediction using a dictionary as input.
        
        Args:
            endpoint_name: Name of the endpoint
            data_dict: Dictionary with features
            params: Optional parameters for the model
            
        Returns:
            Prediction results
        """
        request = PredictionRequest[Dict[str, Any]](
            inputs=data_dict,
            params=params
        )
        
        return self._make_prediction(endpoint_name, request)
    
    def _make_prediction(
        self, 
        endpoint_name: str, 
        request: PredictionRequest[Any]
    ) -> Any:
        """
        Make a prediction using the specified request.
        
        Args:
            endpoint_name: Name of the endpoint
            request: PredictionRequest instance
            
        Returns:
            Prediction results
        """
        url = f"{self.workspace_url}/serving-endpoints/{endpoint_name}/invocations"
        headers = {
            **self._get_auth_headers(),
            'Content-Type': 'application/json',
        }
        
        data = request.to_dict()
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code != 200:
            self.logger.error(f"Prediction failed: {response.text}")
            raise ValueError(f"Prediction failed: {response.text}")
        
        result = response.json()
        
        # Return predictions with proper typing
        if 'predictions' in result:
            return result['predictions']
        else:
            return result
    
    def wait_for_endpoint_ready(
        self, 
        endpoint_name: str, 
        timeout_seconds: int = 300, 
        poll_interval_seconds: int = 10
    ) -> EndpointStatus:
        """
        Wait for an endpoint to be in READY state with timeout.
        
        Args:
            endpoint_name: Name of the endpoint
            timeout_seconds: Maximum time to wait in seconds
            poll_interval_seconds: Time between status checks
            
        Returns:
            Final endpoint status
        """
        self.logger.info(f"Waiting for endpoint {endpoint_name} to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            status = self.get_endpoint_status(endpoint_name)
            
            if status == EndpointStatus.READY:
                self.logger.info(f"Endpoint {endpoint_name} is ready")
                return status
            elif status == EndpointStatus.FAILED:
                self.logger.error(f"Endpoint {endpoint_name} failed")
                return status
            
            self.logger.info(f"Endpoint {endpoint_name} status: {status.value}, waiting...")
            time.sleep(poll_interval_seconds)
        
        self.logger.warning(f"Timed out waiting for endpoint {endpoint_name}")
        return self.get_endpoint_status(endpoint_name)

# Concrete type-specific implementations
class TabularModelClient(ModelServingClient):
    """Strongly-typed client for tabular model endpoints."""
    
    def predict(
        self, 
        endpoint_name: str, 
        features: pd.DataFrame, 
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Make a prediction with a tabular model.
        
        Args:
            endpoint_name: Name of the endpoint
            features: DataFrame with features
            params: Optional parameters for the model
            
        Returns:
            DataFrame with prediction results
        """
        result = self.predict_pandas(endpoint_name, features, params)
        
        # For tabular models, convert result to DataFrame for type safety
        if isinstance(result, list) and all(isinstance(item, dict) for item in result):
            return pd.DataFrame(result)
        elif isinstance(result, dict) and 'predictions' in result:
            return pd.DataFrame(result['predictions'])
        else:
            # Fall back to returning as is with warning
            self.logger.warning(f"Unexpected result format: {type(result)}")
            return pd.DataFrame(result)

class TextGenerationClient(ModelServingClient):
    """Strongly-typed client for text generation model endpoints."""
    
    def generate(
        self, 
        endpoint_name: str, 
        prompts: List[str], 
        params: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate text using an LLM endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            prompts: List of text prompts
            params: Optional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            List of generated text responses
        """
        default_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.95
        }
        
        merged_params = {**default_params, **(params or {})}
        
        input_data = {"inputs": prompts}
        result = self.predict_dict(endpoint_name, input_data, merged_params)
        
        # Extract responses with proper type handling
        if isinstance(result, list):
            return [r.get('text', '') if isinstance(r, dict) else str(r) for r in result]
        elif isinstance(result, dict) and 'predictions' in result:
            predictions = result['predictions']
            if isinstance(predictions, list):
                return [p.get('text', '') if isinstance(p, dict) else str(p) for p in predictions]
        
        # Fall back with warning
        self.logger.warning(f"Unexpected result format: {type(result)}")
        return [str(result)]

class ImageGenerationClient(ModelServingClient):
    """Strongly-typed client for image generation model endpoints."""
    
    def generate_images(
        self, 
        endpoint_name: str, 
        prompts: List[str], 
        params: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate images using a diffusion model endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            prompts: List of text prompts describing images
            params: Optional generation parameters
            
        Returns:
            List of image URLs or Base64-encoded images
        """
        default_params = {
            "num_images_per_prompt": 1,
            "image_width": 512,
            "image_height": 512
        }
        
        merged_params = {**default_params, **(params or {})}
        
        input_data = {"inputs": prompts}
        result = self.predict_dict(endpoint_name, input_data, merged_params)
        
        # Extract image data with type safety
        if isinstance(result, list):
            return [r.get('image', '') if isinstance(r, dict) else str(r) for r in result]
        elif isinstance(result, dict) and 'predictions' in result:
            predictions = result['predictions']
            if isinstance(predictions, list):
                return [p.get('image', '') if isinstance(p, dict) else str(p) for p in predictions]
        
        # Fall back with warning
        self.logger.warning(f"Unexpected result format: {type(result)}")
        return [str(result)]

# Factory function to create strongly-typed clients
def create_model_client(
    client_type: str,
    workspace_url: str,
    credentials: EndpointCredentials,
    logger: Optional[logging.Logger] = None
) -> ModelServingClient:
    """
    Create a type-specific model client based on the specified type.
    
    Args:
        client_type: Type of client ("tabular", "text", "image")
        workspace_url: Databricks workspace URL
        credentials: Authentication credentials
        logger: Optional logger instance
        
    Returns:
        Appropriate typed model client
    """
    if client_type.lower() == "tabular":
        return TabularModelClient(workspace_url, credentials, logger)
    elif client_type.lower() == "text":
        return TextGenerationClient(workspace_url, credentials, logger)
    elif client_type.lower() == "image":
        return ImageGenerationClient(workspace_url, credentials, logger)
    else:
        return ModelServingClient(workspace_url, credentials, logger)
