"""Shared configuration loader for the project."""
import yaml
import os

def load_config(config_path: str = 'src/training/config.yaml') -> dict:
    """Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters as a dictionary.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_class_names(config: dict) -> list:
    """Get class names from the configuration.
    
    Args:
        config (dict): Configuration dictionary containing 'class_names'.

    Returns:
        list: List of class names.
    
    Raises:
        KeyError: If 'class_names' is not found in the configuration.
    """
    if 'class_names' not in config:
        raise KeyError("Configuration must contain 'class_names'")
    
    return config.get('class_names', ['adulterated', 'violent', 'safe'])