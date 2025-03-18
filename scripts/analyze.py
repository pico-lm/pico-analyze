#!/usr/bin/env python3
"""
The main script for running learning dynamics analysis.

Given a metrics config and a trained model, this script will load in the model at different
checkpoints and computed the specified learning dynamics metrics.
"""

import json
import os
from dataclasses import asdict

import click

from src.metrics import BaseComparativeMetric, get_metric
from src.utils.data import get_checkpoint_states, get_training_config
from src.utils.exceptions import InvalidStepError
from src.utils.initialization import (
    CheckpointLocation,
    initialize_config,
    initialize_logging,
    initialize_output_dir,
    initialize_wandb,
)
from src.utils.logging import pretty_print_component_metrics, pretty_print_config


@click.command()
@click.option(
    "--config_path",
    type=str,
    required=True,
    help="Path to the metrics configuration file.",
)
@click.option("--repo_id", type=str, help="Repository ID.")
@click.option("--branch", type=str, help="Branch name.")
@click.option("--run_path", type=str, help="Path to the run directory.")
def main(config_path: str, repo_id: str, branch: str, run_path: str):
    """
    The main function for running learning dynamics analysis. Also note that config_path is a
    required argument, AND either repo_id must be provided or branch and run_path must be provided.
    If this is not specified here, it will raise an error when the checkpoint location is
    initialized.

    Args:
        config_path: str -- the path to the metrics configuration file.  (required)

        repo_id: str -- the repository id.
        branch: str -- the branch name.
        run_path: str -- the path to the run directory.
    """

    # Loads in the metrics config (the config that specifies the metrics to compute)
    metrics_config = initialize_config(config_path)

    # A helper class that stores the checkpoint location (either a local run or a remote run on HF)
    # NOTE: this will raise an error if repo_id is not provided and branch and run_path are not
    # provided.
    checkpoint_location = CheckpointLocation(repo_id, branch, run_path)

    # Loads in the training config (the config that specifies the model architecture, etc.) for the
    # given checkpoint location. NOTE: we use this to automatically determine parts of the model
    # architecture (e.g. the hidden dimension, number of attention heads, etc.)
    training_config = get_training_config(checkpoint_location)

    ############################################################
    #
    # Monitoring Setup (Logging and Wandb)
    #
    ############################################################

    # Set up the output directory
    output_dir = initialize_output_dir(metrics_config, training_config)
    logger = initialize_logging(output_dir)

    # Log the learning dynamics and training configurations to the logger
    logger.info("=" * 80)
    logger.info("Initializing Pico Analysis")
    logger.info("=" * 80)

    pretty_print_config(logger, "Learning Dynamics Config", asdict(metrics_config))
    pretty_print_config(logger, "Training Config", training_config)

    logger.info("=" * 80 + "\n")

    # Set up the wandb run
    if metrics_config.monitoring.save_to_wandb:
        wandb_run = initialize_wandb(metrics_config)

    ############################################################
    #
    # Setting up Metrics
    #
    ############################################################

    metrics = {}

    # Setup all of the metrics
    for metric_config in metrics_config.metrics:
        # Sets up the metric specified in the metrics config
        metric = get_metric(metric_config, training_config)

        # NOTE: if the metric is a comparative metric, we need to set the target checkpoint
        # for the metric.
        if isinstance(metric, BaseComparativeMetric):
            target_checkpoint_states = get_checkpoint_states(
                checkpoint_location=checkpoint_location,
                step=metric_config.target_checkpoint,
                data_split=metric_config.data_split,
            )
            metric.set_target(target_checkpoint_states)

        metrics[metric_config.metric_name] = metric

    ############################################################
    #
    # Computing and Logging Metrics over the checkpoint steps
    #
    ############################################################

    # Computing the metrics for each step
    for step in metrics_config.steps:
        step_directory = os.path.join(output_dir, f"step_{step}")
        os.makedirs(step_directory, exist_ok=True)

        step_metrics = {}

        for metric_name, metric in metrics.items():
            try:
                checkpoint_states = get_checkpoint_states(
                    checkpoint_location=checkpoint_location,
                    step=step,
                    data_split=metric.metric_config.data_split,
                )
            except InvalidStepError:
                # NOTE: this can happen if the step is not available for the given data split;
                # e.g. mostly likely to happen for the last-step of the training run if a metric
                # was not computed on the training data.
                logger.warning(
                    f"Skipping step {step} for metric {metric_name} on split {metric.metric_config.data_split} because the checkpoint does not exist"
                )
                continue

            # NOTE: metric returns a list of dictionaries which corresponds to metric data
            # for each component specified in the metrics config.
            component_metrics_list = metric(checkpoint_states)

            component_metrics_dict = {}
            for component_metrics in component_metrics_list:
                component_metrics_dict.update(component_metrics)

            step_metrics[metric_name] = component_metrics_dict

            # store out the data to the output directory
            with open(
                os.path.join(
                    step_directory,
                    f"{metric_name}_{metric.metric_config.data_split}.json",
                ),
                "w",
            ) as f:
                json.dump(component_metrics_dict, f)

            if metrics_config.monitoring.save_to_wandb:
                # Create a nested dictionary with metric name as prefix
                wandb_formatted_data = {
                    f"{metric_name}_{metric.metric_config.data_split}/{layer}": value
                    for layer, value in component_metrics_dict.items()
                }
                # Add the step information
                wandb_run.log(wandb_formatted_data, step=step)

        # Log out all of the metrics at the current step
        pretty_print_component_metrics(logger, step, step_metrics)


if __name__ == "__main__":
    main()
