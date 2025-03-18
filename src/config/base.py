"""
Base configuration classes for the config system which is composed primarily of metrics that
operate on top of components.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# NOTE: Base class for components of a metrics.


@dataclass
class BaseComponentConfig:
    """
    Base configuration for a component of a model.

    A component can be a single layer, group of layers, activations etc, the choice is arbitrary,
    as long as the given metric defines how to compute the metric for the component.

    Example:

        component_name: ov_circuit # name of the component
        layer_suffixes:
            output_layer: "attention.out_proj" # suffix of the layer to compute the metric for
            value_layer: "attention.v_proj" # suffix of the layer to compute the metric for
        layers: [0,1,2,3,4,5,6,7,8,9,10,11] # layers to compute the metric for
        data_type: "weights" # type of checkpoint data to compute the component for (e.g. "weights", "activations", "gradients")

    """

    component_name: str  # name of the component
    layer_suffixes: (
        str | Dict[str, str]
    )  # suffixes of the layers to compute the metric for
    layers: List[int]  # layers to compute the metric for
    data_type: str = None  # type of checkpoint data to compute the component for (e.g. "weights", "activations", "gradients")


"""
NOTE: Base class for metrics.

There are two types of metrics:
1. BaseMetricConfig: these are metrics that are computed on a single checkpoint; e.g. the 
    norm of a layer at a given checkpoint.
2. BaseComparativeMetricConfig: these are metrics that are computed on a pair of checkpoints; e.g.
    the CKA between two layers at two different checkpoints to measure how similar the two layers
    are between the two checkpoints.
"""


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

        for component_config in self.components:
            if isinstance(component_config, dict):
                _process_components.append(BaseComponentConfig(**component_config))
            else:
                _process_components.append(component_config)

        self.components = _process_components


@dataclass
class BaseComparativeMetricConfig(BaseMetricConfig):
    """
    Base configuration for a comparative metric (which is a subclass of BaseMetricConfig).

    A comparative metric is a metric that is computed on a pair of checkpoints to compare how
    a model's activations or weights change between two different checkpoints.

    Args:
        target_checkpoint: int -- the checkpoint to compare the source checkpoint to.
    """

    target_checkpoint: int = None
