"""
Initialize configuration objects from a YAML file.
"""

import os
import yaml
from dataclasses import fields, is_dataclass

from src.utils.exceptions import InvalidLocationError
from src.config.learning_dynamics import LearningDynamicsConfig

####################
#
# Helper Functions and Classes
#
####################


class CheckpointLocation:
    def __init__(self, repo_id: str, branch: str, run_path: str):
        """
        Initialize a CheckpointLocation object. Used to specify the location of a checkpoint
        which can be either local or remote.
        """
        self.repo_id = repo_id
        self.branch = branch
        self.run_path = run_path

        self._validate_input()

    def _validate_input(self):
        """
        Need to ensure that either the repo_id and branch are specified or the run_path is specified.
        """
        if self.run_path is not None:
            if os.path.exists(self.run_path):
                InvalidLocationError(self.run_path)
            self.is_remote = False
        else:
            if self.repo_id is None or self.branch is None:
                raise InvalidLocationError(self.run_path)
            self.is_remote = True


####################
#
# Configuration Setup
#
####################


def _apply_config_overrides(config, overrides: dict):
    """Recursively apply configuration overrides to a dataclass config object.

    Args:
        config: Base configuration object (must be a dataclass)
        overrides: Dictionary of override values matching config structure

    Returns:
        Modified config object with overrides to the config.
    """
    for field in fields(config):
        field_value = getattr(config, field.name)
        if is_dataclass(field_value):
            _apply_config_overrides(field_value, overrides.get(field.name, {}))
        else:
            if field.name in overrides:
                setattr(config, field.name, overrides[field.name])
    return config


def initialize_config(config_path: str) -> dict:
    """Initialize configuration objects with optional overrides from a YAML file.

    This function initializes the configuration objects with the default values, and then
    applies any overrides from the config_path file.

    Args:
        config_path: Path to a YAML file containing configuration overrides.

    Returns:
        A dictionary containing the initialized configuration objects.
    """
    overrides = yaml.safe_load(open(config_path, "r"))
    config = LearningDynamicsConfig(**overrides)
    return config
