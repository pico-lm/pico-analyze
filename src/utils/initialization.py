"""
Initialize configuration objects from a YAML file.
"""

from datetime import datetime
import logging
import os

# typing imports
from typing import Any, Dict

import wandb
import yaml

from src.config.learning_dynamics import LearningDynamicsConfig
from src.utils.exceptions import InvalidRunLocationError

####################
#
# Monitoring Setup (Logging and Wandb)
#
####################


def initialize_output_dir(config: LearningDynamicsConfig, training_config: Dict[str, Any]) -> str:
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
            training_config["checkpointing"]["run_name"] + "_analysis_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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
    logger = logging.getLogger("pico-analyze")
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


def initialize_wandb(config: LearningDynamicsConfig) -> wandb.sdk.wandb_run.Run:
    """
    Sets up the Wandb run tracker to log out the learning dynamics metrics. Reads in the
    config and training config and initializes a wandb run; if the run already exists, and no
    entity or project is specified in the config, then wandb will print out the metrics
    to the existing run.

    Args:
        config: LearningDynamicsConfig -- the learning dynamics config.

    Returns:
        wandb.sdk.wandb_run.Run -- the wandb run.
    """

    if not config.monitoring.save_to_wandb:
        return None

    # check if there is a wandb entity and project specified in the config
    assert config.monitoring.wandb.entity is not None, "Wandb entity must be specified in the config."
    assert config.monitoring.wandb.project is not None, "Wandb project must be specified in the config."

    entity = config.monitoring.wandb.entity
    project = config.monitoring.wandb.project

    run_name = config.analysis_name

    # initialize the wandb logger
    wandb_run = wandb.init(
        name=run_name,
        project=project,
        entity=entity,
    )

    return wandb_run


def initialize_pico_reporter(config: LearningDynamicsConfig):
    """
    Sets up the Pico Reporter to log out learning dynamics metrics to Pico Labs.

    This function initializes a PicoReporter instance that can be used to log analysis
    metrics to the Pico Labs platform. It requires PICO_API_KEY and PICO_LAB_HASH
    environment variables to be set.

    The reporter can automatically create git commits for each analysis run when auto_commit
    is enabled in the config (pico_report.auto_commit). This captures the exact code state
    and links it to your analysis, allowing you to see code diffs between analysis runs
    in the dashboard.

    Args:
        config: LearningDynamicsConfig -- the learning dynamics config.

    Returns:
        PicoReporter instance or None if save_to_picolabs is False

    Raises:
        ImportError: If pico-report is not installed
        PicoConfigError: If required environment variables are not set
    """
    if not config.monitoring.save_to_picolabs:
        return None

    try:
        from pico_report.integrations import PicoReporter
    except ImportError:
        raise ImportError("pico-report is not installed. Please install it with: " "pip install pico-report")

    # Lab hash can be provided via config or environment variable (PICO_LAB_HASH)
    lab_hash = config.monitoring.pico_report.lab_hash
    if not lab_hash:
        lab_hash = os.getenv("PICO_LAB_HASH", "")

    assert (
        lab_hash is not None and lab_hash != ""
    ), "Lab hash must be provided via config (pico_report.lab_hash) or PICO_LAB_HASH environment variable."

    # If no experiment name is specified, use the analysis name
    experiment_name = config.monitoring.pico_report.experiment_name
    if not experiment_name: 
        experiment_name = config.analysis_name

    # Get auto_commit setting from config
    auto_commit = config.monitoring.pico_report.auto_commit

    # Create PicoReporter instance with auto_commit setting
    pico_reporter = PicoReporter(lab_hash=lab_hash, experiment_name=experiment_name, auto_commit=auto_commit)

    # Setup experiment
    pico_reporter.setup_experiment(
        experiment_name=experiment_name,
        description=f"Learning dynamics analysis: {config.analysis_name}",
    )

    return pico_reporter


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

        Raises:
            InvalidRunLocationError: If the run_path is not specified and the repo_id and branch are not specified.
        """
        if self.run_path is not None:
            if not os.path.exists(self.run_path):
                raise InvalidRunLocationError()
            self.is_remote = False
        else:
            if self.repo_id is None or self.branch is None:
                raise InvalidRunLocationError()
            self.is_remote = True


####################
#
# Configuration Setup
#
####################


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
