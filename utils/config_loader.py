# utils/config_loader.py
import yaml
import os

_CONFIG_CACHE = {}

def get_config(config_name: str) -> dict:
    """
    Loads and parses a YAML configuration file from the configs/ directory.
    Uses caching so each file is read from disk only once per process.
    
    Args:
        config_name (str): Name of the config file without extension (e.g., 'env', 'sac').
    
    Returns:
        dict: The parsed YAML contents.
    """
    if config_name in _CONFIG_CACHE:
        return _CONFIG_CACHE[config_name]
        
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'configs', f"{config_name}.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    _CONFIG_CACHE[config_name] = config or {}
    return _CONFIG_CACHE[config_name]
