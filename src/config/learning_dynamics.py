"""
Configuration classes for learning dynamics analysis.
"""

from dataclasses import dataclass, field
from typing import List

from src.config.metrics import (
    BaseMetricConfig,
    BaseComparativeMetricConfig,
    METRIC_CONFIG_REGISTRY,
)


@dataclass
class LearningDynamicsConfig:
    """
    Root configuration for learning dynamics metrics. Contains a list of metrics to compute,
    and the steps to compute them for.

    Attributes:
        metrics: List of metric configurations
        steps: List of steps to compute metrics for
    """

    metrics: List[BaseMetricConfig | BaseComparativeMetricConfig] = field(
        default_factory=list
    )
    steps: List[int] = field(default_factory=list)

    def __post_init__(self):
        """
        Post-initialization method to convert metric dictionaries to proper config objects. Used
        for loading in metrics from a yaml file where the metrics are specified as dictionaries.

        Example yaml file:
        metrics:
            - metric_name: cka
              target_checkpoint: 1000
              [...]
            - metric_name: gradient_similarity
              [...]

        This will be converted to the following config object:
        LearningDynamicsConfig(
            metrics=[
                CKAConfig(metric_name="cka", target_checkpoint=1000, ...),
                GradientSimilarityConfig(metric_name="gradient_similarity", ...)
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
            self.steps = range(
                self.steps["start"], self.steps["end"], self.steps["step"]
            )
        elif isinstance(self.steps, list):
            self.steps = [int(step) for step in self.steps]
        else:
            raise ValueError("steps must be a list of integers or a StepRangeConfig")
