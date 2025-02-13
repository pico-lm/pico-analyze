"""
Configuration classes for learning dynamics analysis.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import List, Dict, Any

#####################
# Metrics
#####################

# Abstract Metric Classes

# NOTE: There are two types of metrics:
# 1. BaseMetricConfig: these are metrics that are computed on a single checkpoint
# 2. BaseComparativeMetricConfig: these are metrics that are computed on a pair of checkpoints

# -----------------
# Base Metrics
# -----------------


@dataclass
class BaseMetricConfig(ABC):
    """
    Base configuration for a metric. All metrics should implement this class.
    """

    metric_name: str
    data_split: str


@dataclass
class BaseComparativeMetricConfig(ABC):
    """
    Base configuration for a metric.
    """

    target_checkpoint: int
    metric_name: str
    data_split: str


# -----------------
# Metrics
# -----------------


@dataclass
class CKAConfig(BaseComparativeMetricConfig):
    """
    Configuration for the CKA metric. NOTE that because this is a comparative metric,
    we need to specify the target_checkpoint.
    """

    metric_name: str = "cka"
    data_split: str = "val"

    # Specify what components to compute the CKA for; to do so, we specify a list of dictionaries,
    # where each dictionary specifies the component name and the layer suffixes and layers to compute
    # the CKA for.

    # For example, if we want to compute the CKA for the OV circuit and the MLP, we would specify the following:

    # components:
    #   - name: ov_circuit
    #       layer_suffixes:
    #         output_layer: "attention.out_proj"
    #         value_layer: "attention.v_proj"
    #       layers: [0,1,2,3,4,5,6,7,8,9,10,11]
    #   - name: mlp
    #       layer_suffixes:
    #         mlp: "swiglu.w_2"
    #       layers: [0,1,2,3,4,5,6,7,8,9,10,11]

    components: List[Dict[str, Any]] = field(default_factory=list)


# TODO: add gradient similarity metric
# @dataclass
# class GradientSimilarityConfig(BaseMetricConfig):
#     """
#     Configuration for gradient similarity metric.
#     """
#     metric_name: str = "gradient_similarity"

# TODO: add other metrics


#####################
# Root Metrics Config
#####################


@dataclass
class LearningDynamicsConfig:
    """
    Configures the learning dynamic metrics to be computed.

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
            - metric_name: gradient_similarity

        This will be converted to the following config object:
        LearningDynamicsConfig(
            metrics=[
                CKAConfig(metric_name="cka", target_checkpoint=1000),
                GradientSimilarityConfig(metric_name="gradient_similarity")
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

                    if metric_name == "cka":
                        processed_metrics.append(CKAConfig(**metric))
                    elif metric_name == "gradient_similarity":
                        # processed_metrics.append(GradientSimilarityConfig(**metric))
                        pass
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
