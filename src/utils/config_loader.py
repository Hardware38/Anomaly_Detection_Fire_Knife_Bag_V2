import yaml
import os
from pathlib import Path


class ConfigLoader:
    """Load and manage configuration settings."""

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key_path, default=None):
        """
        Get configuration value using dot notation.
        Example: get('fire_detection.model_name')
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_fire_config(self):
        """Get fire detection configuration."""
        return self.config.get('fire_detection', {})

    def get_weapon_config(self):
        """Get weapon detection configuration."""
        return self.config.get('weapon_detection', {})

    def get_railway_config(self):
        """Get railway detection configuration."""
        return self.config.get('railway_detection', {})

    def get_general_config(self):
        """Get general configuration."""
        return self.config.get('general', {})


if __name__ == "__main__":
    config = ConfigLoader()
    print("Fire Detection Config:", config.get_fire_config())
    print("Model Name:", config.get('fire_detection.model_name'))
