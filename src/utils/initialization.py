"""
Initialize configuration objects from a YAML file.
"""

import os
import logging
import yaml
import wandb
from datetime import datetime
from dataclasses import fields, is_dataclass

from src.utils.exceptions import InvalidLocationError
from src.config.learning_dynamics import LearningDynamicsConfig

# typing imports
from typing import Dict, Any

####################
#
# Monitoring Setup (Logging and Wandb)
#
####################


def initialize_output_dir(
    config: LearningDynamicsConfig, training_config: Dict[str, Any]
) -> str:
    """
    Creates the output directory for the analysis. If no analysis name is specified, we will use
    the run name and the current date and time as a unique identifier.

    Args:
        config: LearningDynamicsConfig -- the learning dynamics config.
        training_config: Dict[str, Any] -- the training config.

    Returns:
        str -- the output directory.
    """

    _analysis_name = config.analysis_name
    if _analysis_name is None or _analysis_name == "":
        # if no analysis name is specified, use the run name and the current date and time
        # as a unique identifier
        _analysis_name = (
            training_config["checkpointing"]["run_name"]
            + "_analysis_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    config.analysis_name = _analysis_name

    analysis_dir = os.path.join(config.monitoring.output_dir, _analysis_name)
    os.makedirs(analysis_dir, exist_ok=True)
    return analysis_dir


def initialize_logging(analysis_dir: str) -> logging.Logger:
    """
    Sets up the logging for the analysis. The logs are saved to the analysis directory.

    Args:
        analysis_dir: str -- the analysis directory to save the logs to

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("pico-analysis")
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(analysis_dir, "analysis.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def initialize_wandb(
    config: LearningDynamicsConfig, training_config: Dict[str, Any]
) -> wandb.sdk.wandb_run.Run:
    """
    Sets up the Wandb run tracker to log out the learning dynamics metrics. Reads in the
    config and training config and initializes a wandb run; if the run already exists, and no
    entity or project is specified in the config, then wandb will print out the metrics
    to the existing run.

    The prefered work-flow is to just either specify the entity and project in the config or
    not at all and let the analysis save to the original run.

    Args:
        config: LearningDynamicsConfig -- the learning dynamics config.
        training_config: Dict[str, Any] -- the training config.

    Returns:
        wandb.sdk.wandb_run.Run -- the wandb run.
    """

    assert (
        config.monitoring.save_to_wandb
    ), "Wandb is not enabled, so we cannot setup the wandb logger."

    # If there is no entity or project specified in the config, we will save the analysis to the
    # original run.
    save_to_original_run = True

    # check if there is a wandb entity and project specified in the config
    if config.monitoring.wandb.entity is not None:
        entity = config.monitoring.wandb.entity

        save_to_original_run = False
    else:
        try:
            entity = training_config["monitoring"]["experiment_tracker"]["wandb_entity"]
        except KeyError:
            raise ValueError(
                "Wandb entity must be specified in the config or training config."
            )

    if config.monitoring.wandb.project is not None:
        project = config.monitoring.wandb.project

        save_to_original_run = False
    else:
        try:
            project = training_config["monitoring"]["experiment_tracker"][
                "wandb_project"
            ]
        except KeyError:
            raise ValueError(
                "Wandb project must be specified in the config or training config."
            )

    if save_to_original_run:
        # get the run_name from the training config
        run_name = training_config["checkpointing"]["run_name"]
        # get the run_id from wandb
        previous_runs = wandb.Api().runs(
            path=f"{entity}/{project}",
            filters={"display_name": run_name},
        )
        try:
            _run_id = previous_runs[0].id
        except ValueError:
            # NOTE if we can't find a previous run, then this will create a new run with the same
            # name as the training run, but will show up as a new run in wandb.
            _run_id = None
    else:
        # NOTE: in the output_dir we create a unique analysis name if none is specified
        run_name = config.analysis_name
        _run_id = None

    # initialize the wandb logger
    wandb_run = wandb.init(
        name=run_name,
        project=project,
        id=_run_id,
        entity=entity,
    )

    return wandb_run


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
