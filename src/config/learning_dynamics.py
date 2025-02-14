"""
Configuration classes for learning dynamics analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict

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
class BaseComponentConfig:
    """
    Base configuration for a component of a model.

    A component can be a single layer, group of layers, activations etc, the choice is arbitrary,
    as long as the given metric defines how to compute the metric for the component.

    Example:

        name: ov_circuit # name of the component
        layer_suffixes:
            output_layer: "attention.out_proj" # suffix of the layer to compute the metric for
            value_layer: "attention.v_proj" # suffix of the layer to compute the metric for
        layers: [0,1,2,3,4,5,6,7,8,9,10,11] # layers to compute the metric for

    Args:
        name: str -- the name of the component.
        layer_suffixes: Dict[str, str] -- maps the names of layers that make up a component
            to the suffixes of those layers in the model.
        layers: List[int] -- the layers of the model to compute the metric for.
    """

    name: str = None
    layer_suffixes: Dict[str, str] = None
    layers: List[int] = None


@dataclass
class BaseMetricConfig:
    """
    Base configuration for a metric. All metrics should implement this class. Requires
    specifying the components to compute the metric for, the metric name, and the data split.

    Args:
        components: List[BaseComponentConfig] -- the components to compute the metric for.
        metric_name: str -- the name of the metric.
        data_split: str -- the data split to compute the metric for (e.g. "train", "val", "test").

    """

    components: List[BaseComponentConfig] = field(default_factory=list)

    metric_name: str = None
    data_split: str = None

    def __post_init__(self):
        """
        Post-initialization method to convert yaml dictionaries of components to proper
        BaseComponentConfig objects.
        """
        for component in self.components:
            if isinstance(component, dict):
                self.components.append(BaseComponentConfig(**component))


@dataclass
class BaseComparativeMetricConfig(BaseMetricConfig):
    """
    Base configuration for a comparative metric (subclass of BaseMetricConfig).

    A comparative metric is a metric that is computed on a pair of checkpoints to compare how
    a model's activations or weights change between two different checkpoints.

    Args:
        target_checkpoint: int -- the checkpoint to compare the source checkpoint to.
    """

    target_checkpoint: int = None


# -----------------
# Metrics
# -----------------


@dataclass
class NormMetricConfig(BaseMetricConfig):
    """
    Configuration for a norm metric.

    Args:
        data_type: str -- the type of data to compute the norm for (e.g. "weights", "activations", "gradients").
    """

    metric_name: str = "norm"

    # NOTE: used to specify what type of norm to compute:
    #       options are "Frobenius", "spectral", "max"
    norm_type: str = None

    # NOTE: used to specify what type of norm to compute;
    #       options are "weights", "activations", "gradients"
    data_type: str = None


@dataclass
class PERConfig(BaseMetricConfig):
    """
    Configuration for the Proportional Effective Rank (PER) metric.
    The PER is a metric that measures the effective rank of a matrix, and is defined in:
        Tending Towards Stability: Convergence Challenges in Small Language Models
        https://aclanthology.org/2024.findings-emnlp.187/

    Args:
        data_type: str -- the type of data to compute the PER for (e.g. "weights", "activations", "gradients").
    """

    metric_name: str = "per"

    # NOTE: used to specify what type of norm to compute;
    #       options are "weights", "gradients"
    data_type: str = None


# -----------------
# Comparative Metrics
# -----------------


@dataclass
class CKAConfig(BaseComparativeMetricConfig):
    """
    Configuration for the CKA metric. NOTE that because this is a comparative metric,
    we need to specify the target_checkpoint.
    """

    metric_name: str = "cka"


#####################
# Root Metrics Config
#####################


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
