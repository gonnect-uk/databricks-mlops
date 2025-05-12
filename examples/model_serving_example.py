#!/usr/bin/env python
"""
Example demonstrating the type-safe model serving client for Databricks endpoints.

This example shows how to use the strongly-typed client to interact with
Databricks model serving endpoints using Pydantic models and proper type validation.
"""
import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from databricks_mlops.utils.model_serving import (AuthType, EndpointCredentials,
                                               TabularModelClient, TextGenerationClient,
                                               ImageGenerationClient, create_model_client)
from databricks_mlops.utils.logging import setup_logger


class ServingEndpointExample:
    """Example class demonstrating how to use the model serving clients."""
    
    def __init__(self):
        """Initialize the example with logger."""
        self.logger = setup_logger("serving_example")
    
    def run_tabular_model_example(self, workspace_url: str, endpoint_name: str, token: str) -> None:
        """
        Run an example for a tabular model serving endpoint.
        
        Args:
            workspace_url: Databricks workspace URL
            endpoint_name: Serving endpoint name
            token: Authentication token
        """
        self.logger.info("Running tabular model serving example")
        
        # Create strongly-typed credentials
        credentials = EndpointCredentials(
            auth_type=AuthType.TOKEN,
            token=token
        )
        
        # Create strongly-typed client
        client = TabularModelClient(
            workspace_url=workspace_url,
            credentials=credentials,
            logger=self.logger
        )
        
        # Create sample data - this would be your features for inference
        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature3': [10, 20, 30, 40, 50],
            'categorical': ['A', 'B', 'A', 'C', 'B']
        })
        
        self.logger.info(f"Making prediction with data shape: {data.shape}")
        
        try:
            # Check if endpoint is ready
            status = client.wait_for_endpoint_ready(
                endpoint_name=endpoint_name,
                timeout_seconds=30  # Short timeout for the example
            )
            
            self.logger.info(f"Endpoint status: {status}")
            
            # Make prediction with type safety
            predictions = client.predict(
                endpoint_name=endpoint_name,
                features=data
            )
            
            self.logger.info(f"Received predictions with shape: {predictions.shape}")
            self.logger.info(f"Predictions:\n{predictions.head()}")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in tabular model example: {str(e)}")
            raise
    
    def run_text_generation_example(self, workspace_url: str, endpoint_name: str, token: str) -> None:
        """
        Run an example for a text generation model serving endpoint.
        
        Args:
            workspace_url: Databricks workspace URL
            endpoint_name: Serving endpoint name
            token: Authentication token
        """
        self.logger.info("Running text generation model serving example")
        
        # Create strongly-typed credentials
        credentials = EndpointCredentials(
            auth_type=AuthType.TOKEN,
            token=token
        )
        
        # Create strongly-typed client
        client = TextGenerationClient(
            workspace_url=workspace_url,
            credentials=credentials,
            logger=self.logger
        )
        
        # Sample prompts
        prompts = [
            "Explain the concept of MLOps in simple terms.",
            "Write a short poem about machine learning."
        ]
        
        # Generation parameters
        params = {
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 0.95
        }
        
        self.logger.info(f"Generating text for {len(prompts)} prompts")
        
        try:
            # Check if endpoint is ready
            status = client.wait_for_endpoint_ready(
                endpoint_name=endpoint_name,
                timeout_seconds=30  # Short timeout for the example
            )
            
            self.logger.info(f"Endpoint status: {status}")
            
            # Generate text with type safety
            responses = client.generate(
                endpoint_name=endpoint_name,
                prompts=prompts,
                params=params
            )
            
            # Print responses
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                self.logger.info(f"\nPrompt {i+1}: {prompt}")
                self.logger.info(f"Response: {response}")
            
            return responses
            
        except Exception as e:
            self.logger.error(f"Error in text generation example: {str(e)}")
            raise
    
    def run_demo_with_mocks(self) -> None:
        """
        Run a demonstration using mock data when real credentials aren't available.
        """
        self.logger.info("Running demonstration with mock data")
        
        # Create sample data for tabular model
        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature3': [10, 20, 30, 40, 50],
            'categorical': ['A', 'B', 'A', 'C', 'B']
        })
        
        # Mock predictions for tabular model
        predictions = pd.DataFrame({
            'prediction': [0, 1, 0, 1, 1],
            'probability': [0.2, 0.8, 0.3, 0.9, 0.7]
        })
        
        self.logger.info("=== Tabular Model Endpoint Example ===")
        self.logger.info(f"Input features:\n{data.head()}")
        self.logger.info(f"Predicted results:\n{predictions}")
        
        # Mock text generation
        prompts = [
            "Explain the concept of MLOps in simple terms.",
            "Write a short poem about machine learning."
        ]
        
        responses = [
            "MLOps is like having a reliable assembly line for your machine learning models. It helps you build, test, deploy, and monitor your models consistently, just like how a factory makes sure each product meets quality standards before it reaches customers.",
            
            "Silicon thoughts in layers deep,\nPatterns learned while humans sleep.\nNumbers dance and weights adjust,\nIn learning's glow, decisions trust."
        ]
        
        self.logger.info("\n=== Text Generation Endpoint Example ===")
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            self.logger.info(f"\nPrompt {i+1}: {prompt}")
            self.logger.info(f"Response: {response}")
        
        self.logger.info("\nThis example demonstrates the strongly-typed approach for accessing Databricks serving endpoints.")
        self.logger.info("In a real scenario, you would provide actual workspace URL, endpoint name, and authentication token.")


def main() -> None:
    """Run the model serving examples."""
    example = ServingEndpointExample()
    
    # Check if real credentials are provided as environment variables
    workspace_url = os.environ.get("DATABRICKS_WORKSPACE_URL")
    token = os.environ.get("DATABRICKS_TOKEN")
    tabular_endpoint = os.environ.get("DATABRICKS_TABULAR_ENDPOINT")
    
    if workspace_url and token and tabular_endpoint:
        # Run with real credentials
        print("Running with actual Databricks credentials")
        example.run_tabular_model_example(workspace_url, tabular_endpoint, token)
    else:
        # Run the mock example
        print("Running with mock data (no Databricks credentials provided)")
        example.run_demo_with_mocks()


if __name__ == "__main__":
    main()
