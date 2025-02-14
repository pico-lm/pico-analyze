"""
Configuration classes for metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict


METRIC_CONFIG_REGISTRY = {}


def register_metric_config(metric_name: str):
    """
    Decorator to register a metric config class with the MetricConfigRegistry.
    """

    def decorator(cls):
        METRIC_CONFIG_REGISTRY[metric_name] = cls
        return cls

    return decorator


#####################
# Metrics
#####################

# NOTE: There are two types of metrics:
# 1. BaseMetricConfig: these are metrics that are computed on a single checkpoint
# 2. BaseComparativeMetricConfig: these are metrics that are computed on a pair of checkpoints

# -----------------
# Abstract Metric Classes
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
    layer_suffixes: str | Dict[str, str] = None
    layers: List[int] = None


@dataclass
class BaseMetricConfig:
    """
    Base configuration for a metric. All metrics should implement this class. Requires
    specifying the components to compute the metric for, the metric name, and the data split.

    Args:
        metric_name: str -- the name of the metric.
        components: List[BaseComponentConfig] -- the components to compute the metric for.
        data_split: str -- the data split to compute the metric for (e.g. "train", "val", "test").

    """

    metric_name: str = None
    components: List[BaseComponentConfig] = field(default_factory=list)
    data_split: str = None

    def __post_init__(self):
        """
        Post-initialization method to convert yaml dictionaries of components to proper
        BaseComponentConfig objects.
        """
        _process_components = []

        for component in self.components:
            if isinstance(component, dict):
                _process_components.append(BaseComponentConfig(**component))
            else:
                _process_components.append(component)

        self.components = _process_components


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
@register_metric_config("norm")
class NormMetricConfig(BaseMetricConfig):
    """
    Configuration for a norm metric.

    Args:
        data_type: str -- the type of data to compute the norm for (e.g. "weights", "activations", "gradients").
    """

    # NOTE: used to specify what type of norm to compute:
    #       options are "Frobenius", "spectral", "max"
    norm_type: str = None

    # NOTE: used to specify what type of norm to compute;
    #       options are "weights", "activations", "gradients"
    data_type: str = None


@dataclass
@register_metric_config("per")
class PERConfig(BaseMetricConfig):
    """
    Configuration for the Proportional Effective Rank (PER) metric.
    The PER is a metric that measures the effective rank of a matrix, and is defined in:
        Tending Towards Stability: Convergence Challenges in Small Language Models
        https://aclanthology.org/2024.findings-emnlp.187/

    Args:
        data_type: str -- the type of data to compute the PER for (e.g. "weights", "activations", "gradients").
    """

    # NOTE: used to specify what type of norm to compute;
    #       options are "weights", "gradients"
    data_type: str = None


# -----------------
# Comparative Metrics
# -----------------


@dataclass
@register_metric_config("cka")
class CKAConfig(BaseComparativeMetricConfig):
    """
    Configuration for the CKA metric. NOTE that because this is a comparative metric,
    we need to specify the target_checkpoint.
    """

    ...
