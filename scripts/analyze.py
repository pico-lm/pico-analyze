"""
The main script for running learning dynamics analysis.

Given a metrics config and a trained model, this script will load in the model at different
checkpoints and computed the specified learning dynamics metrics.
"""

from src.utils.data import get_learning_dynamics_data, get_training_config
from src.utils.initialization import initialize_config, CheckpointLocation
from src.metrics import get_metric, BaseComparativeMetric

import click


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
    # Loads in the metrics config (the config that specifies the metrics to compute)
    metrics_config = initialize_config(config_path)

    # A helper class that stores the checkpoint location (either a local run or a remote run on HF)
    checkpoint_location = CheckpointLocation(repo_id, branch, run_path)

    # Loads in the training config (the config that specifies the model architecture, etc.) for the
    # given checkpoint location. NOTE: we use this to automatically determine parts of the model
    # architecture (e.g. the hidden dimension, number of attention heads, etc.)
    training_config = get_training_config(checkpoint_location)

    metrics = {}

    # Setup all of the metrics
    for metric_config in metrics_config.metrics:
        # Sets up the metric specified in the metrics config
        metric = get_metric(metric_config, training_config)

        # NOTE: if the metric is a comparative metric, we need to set the target checkpoint
        # for the metric.
        if isinstance(metric, BaseComparativeMetric):
            target_data = get_learning_dynamics_data(
                checkpoint_location=checkpoint_location,
                step=metric_config.target_checkpoint,
                data_split=metric_config.data_split,
            )
            metric.set_target(target_data)

        metrics[metric_config.metric_name] = metric

    # Computing the metrics for each step
    for step in metrics_config.steps:
        step_metric_data = {}

        for metric_name, metric in metrics.items():
            data = get_learning_dynamics_data(
                checkpoint_location=checkpoint_location,
                step=step,
                data_split=metric_config.data_split,
            )

            metric_data = metric.compute(data)

            step_metric_data[metric_name] = metric_data

        # At the end of each step we sve out the data for each of the different metrics
        # TODO: Save out step_metric_data


if __name__ == "__main__":
    main()
