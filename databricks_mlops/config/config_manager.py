"""
Configuration management system with strong typing via Pydantic.
"""
import os
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union, cast

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, create_model, field_validator

from databricks_mlops.models.config import (DataConfig, DeploymentConfig,
                                           FeatureConfig, ModelConfig,
                                           MonitoringConfig, PipelineConfig)
from databricks_mlops.utils.logging import LogLevel, setup_logger

# Type variable for configuration models
T = TypeVar('T', bound=BaseModel)

# Set up logger
logger = setup_logger("config_manager", LogLevel.INFO)


class ConfigManagerError(Exception):
    """Exception raised for errors in the ConfigManager."""
    pass


class EnvVarSubstitutor:
    """Utility for substituting environment variables in configuration values."""
    
    @staticmethod
    def _replace_env_vars(value: str) -> str:
        """
        Replace environment variable references in a string with their values.
        
        Args:
            value: The string potentially containing env var references like ${VAR_NAME}
            
        Returns:
            The string with env vars replaced with their values
        """
        import re
        
        # Pattern to match ${VAR_NAME} or $VAR_NAME
        pattern = r'\${([^}]+)}|\$([A-Za-z0-9_]+)'
        
        def _replace_match(match):
            # Extract the variable name from either group 1 or group 2
            var_name = match.group(1) if match.group(1) else match.group(2)
            # Get the environment variable value, defaulting to an empty string
            return os.environ.get(var_name, '')
        
        # Replace all matched patterns
        return re.sub(pattern, _replace_match, value)
    
    @classmethod
    def substitute_env_vars(cls, data: Any) -> Any:
        """
        Recursively substitute environment variables in configuration data.
        
        Args:
            data: Configuration data (dict, list, or primitive value)
            
        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(data, str):
            return cls._replace_env_vars(data)
        elif isinstance(data, dict):
            return {k: cls.substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.substitute_env_vars(item) for item in data]
        else:
            # Return other types (int, float, bool, None) unchanged
            return data


class ConfigManager(Generic[T]):
    """
    Manages configuration with strong typing through Pydantic models.
    
    This class handles loading configurations from YAML files or dictionaries
    and validating them using Pydantic models.
    """
    
    def __init__(self, config_model: Type[T], env_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_model: Pydantic model class used for configuration validation
            env_file: Optional path to a .env file to load environment variables from
        """
        self.config_model = config_model
        self.config: Optional[T] = None
        
        # Load environment variables if env_file is provided
        if env_file and os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
    
    def load_from_yaml(self, yaml_path: str, env_substitution: bool = True) -> T:
        """
        Load and validate configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            env_substitution: Whether to substitute environment variables in the config
            
        Returns:
            Validated configuration object
            
        Raises:
            ConfigManagerError: If the file cannot be loaded or validation fails
        """
        try:
            if not os.path.exists(yaml_path):
                raise ConfigManagerError(f"Configuration file not found: {yaml_path}")
            
            with open(yaml_path, 'r') as yaml_file:
                config_dict = yaml.safe_load(yaml_file)
                
            if env_substitution:
                config_dict = EnvVarSubstitutor.substitute_env_vars(config_dict)
                
            return self.load_from_dict(config_dict)
            
        except yaml.YAMLError as e:
            error_msg = f"Error parsing YAML file {yaml_path}: {str(e)}"
            logger.error(error_msg)
            raise ConfigManagerError(error_msg) from e
        except Exception as e:
            error_msg = f"Error loading configuration from {yaml_path}: {str(e)}"
            logger.error(error_msg)
            raise ConfigManagerError(error_msg) from e
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> T:
        """
        Load and validate configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            Validated configuration object
            
        Raises:
            ConfigManagerError: If validation fails
        """
        try:
            # Create a new instance of the config model with the provided dictionary
            config = self.config_model.model_validate(config_dict)
            self.config = config
            logger.info(f"Successfully loaded and validated configuration of type {self.config_model.__name__}")
            return config
            
        except Exception as e:
            error_msg = f"Error validating configuration with model {self.config_model.__name__}: {str(e)}"
            logger.error(error_msg)
            raise ConfigManagerError(error_msg) from e
    
    def get_config(self) -> T:
        """
        Get the current configuration.
        
        Returns:
            The current configuration
            
        Raises:
            ConfigManagerError: If no configuration has been loaded
        """
        if self.config is None:
            raise ConfigManagerError("No configuration has been loaded")
        return self.config


class MLOpsConfigManager:
    """
    Configuration manager specifically for MLOps configurations.
    
    This class provides a factory for creating type-specific configuration
    managers for different MLOps pipeline components.
    """
    
    @staticmethod
    def create_data_config_manager(env_file: Optional[str] = None) -> ConfigManager[DataConfig]:
        """Create a configuration manager for data pipelines."""
        return ConfigManager(DataConfig, env_file)
    
    @staticmethod
    def create_feature_config_manager(env_file: Optional[str] = None) -> ConfigManager[FeatureConfig]:
        """Create a configuration manager for feature engineering pipelines."""
        return ConfigManager(FeatureConfig, env_file)
    
    @staticmethod
    def create_model_config_manager(env_file: Optional[str] = None) -> ConfigManager[ModelConfig]:
        """Create a configuration manager for model training pipelines."""
        return ConfigManager(ModelConfig, env_file)
    
    @staticmethod
    def create_deployment_config_manager(env_file: Optional[str] = None) -> ConfigManager[DeploymentConfig]:
        """Create a configuration manager for model deployment pipelines."""
        return ConfigManager(DeploymentConfig, env_file)
    
    @staticmethod
    def create_monitoring_config_manager(env_file: Optional[str] = None) -> ConfigManager[MonitoringConfig]:
        """Create a configuration manager for model monitoring pipelines."""
        return ConfigManager(MonitoringConfig, env_file)
    
    @staticmethod
    def create_pipeline_config_manager(env_file: Optional[str] = None) -> ConfigManager[PipelineConfig]:
        """Create a configuration manager for general pipelines."""
        return ConfigManager(PipelineConfig, env_file)
    
    @staticmethod
    def load_pipeline_config(yaml_path: str, env_file: Optional[str] = None) -> PipelineConfig:
        """
        Load a pipeline configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            env_file: Optional path to a .env file
            
        Returns:
            The pipeline configuration
        """
        config_manager = MLOpsConfigManager.create_pipeline_config_manager(env_file)
        return config_manager.load_from_yaml(yaml_path)
