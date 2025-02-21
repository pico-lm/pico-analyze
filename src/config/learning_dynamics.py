# ruff: noqa: F403

"""
Configuration classes for learning dynamics analysis.
"""

from dataclasses import dataclass, field
from typing import List

from src.config.metrics import *

from src.config._registry import METRIC_CONFIG_REGISTRY

from src.config._base import (
    BaseMetricConfig,
    BaseComparativeMetricConfig,
)

from src.config.monitoring import MonitoringConfig


@dataclass
class LearningDynamicsConfig:
    """
    Root configuration for specifying what learning dynamics metrics to compute, and at which
    checkpoint steps to compute them for. Metrics can be single-checkpoint or comparative metrics,
    and are computed on components of the model.
    """

    # Name for the analysis, used to store/bookkeep the analysis results
    analysis_name: str = None

    metrics: List[BaseMetricConfig | BaseComparativeMetricConfig] = field(
        default_factory=list
    )
    steps: List[int] = field(default_factory=list)

    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    def __post_init__(self):
        """
        Post-initialization method to convert metric dictionaries to proper config objects. Used
        for loading in metrics from a yaml file where the metrics are specified as dictionaries.

        Example yaml file:
        metrics:
            - metric_name: cka
              target_checkpoint: 1000
              data_split: "val"
              components:
                - component_name: simple
                  layer_suffixes: "swiglu.w_2"
                  layers: [0,1,2,3,4,5,6,7,8,9,10,11]
                - component_name: ov_circuit
                  layer_suffixes:
                    output_layer: "attention.o_proj"
                    value_layer: "attention.v_proj"
                  layers: [0,1,2,3,4,5,6,7,8,9,10,11]
            - metric_name: norm
              data_split: "train"
              components:
                - component_name: simple
                  layer_suffixes: "swiglu.w_2"
                  layers: [0,1,2,3,4,5,6,7,8,9,10,11]

        This will be converted to the following config object:
        LearningDynamicsConfig(
            metrics=[
                CKAConfig(metric_name="cka", target_checkpoint=1000, ...),
                NormConfig(metric_name="norm", data_split="train", ...)
            ]
        )
        """
        # Convert metric dictionaries to proper config objects
        if isinstance(self.metrics, list):
            processed_metrics = []
            for metric in self.metrics:
                if isinstance(metric, dict):
                    metric_name = metric.get("metric_name")
                    if metric_name is None:
                        raise ValueError(
                            "metric_name must be specified for each metric"
                        )

                    if metric_name in METRIC_CONFIG_REGISTRY:
                        processed_metrics.append(
                            METRIC_CONFIG_REGISTRY[metric_name](**metric)
                        )
                    else:
                        raise ValueError(f"Unknown metric_name: {metric_name}")
                else:
                    processed_metrics.append(metric)
            self.metrics = processed_metrics

        if isinstance(self.steps, dict):
            self.steps = list(
                range(self.steps["start"], self.steps["end"], self.steps["step"])
            )
        elif isinstance(self.steps, list):
            self.steps = [int(step) for step in self.steps]
        else:
            raise ValueError("steps must be a list of integers or a StepRangeConfig")

        self.monitoring = MonitoringConfig(**self.monitoring)
