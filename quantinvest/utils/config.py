"""
Configuration management module.

This module provides configuration loading and management
utilities for the quantitative investment framework.
"""

import yaml
import os
from typing import Dict, Any, Optional


class Config:
    """
    Configuration manager for the quantitative investment framework.
    
    Loads and manages configuration from YAML files with
    environment variable support.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file
        """
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
        
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations."""
        possible_paths = [
            "config.yaml",
            "config.yml", 
            os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        raise FileNotFoundError("No configuration file found")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Replace environment variables
        return self._substitute_env_vars(config)
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """Substitute environment variables in configuration."""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Parameters:
        -----------
        key : str
            Configuration key (supports dot notation)
        default : Any
            Default value if key not found
            
        Returns:
        --------
        Any
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_data_path(self, data_key: str) -> str:
        """Get full path to data file."""
        root_path = self.get('data.root_path', './data')
        file_path = self.get(f'data.{data_key}')
        
        if file_path is None:
            raise ValueError(f"Data key '{data_key}' not found in configuration")
            
        return os.path.join(root_path, file_path)