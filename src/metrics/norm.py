"""
Norm metrics compute the norm of weights, activations, gradients, etc.
"""

from functools import partial

import torch

from src.config.base import BaseComponentConfig
from src.config.learning_dynamics import BaseMetricConfig
from src.metrics._registry import register_metric
from src.metrics.base import BaseMetric


@register_metric("norm")
class NormMetric(BaseMetric):
    """
    Base class for norm metrics; i.e. metrics that compute the norm of some component data.
    """

    def __init__(self, metric_config: BaseMetricConfig, *args):
        super().__init__(metric_config, *args)

        # NOTE: We use the torch.norm function to compute the norm of the data.
        if self.metric_config.norm_type == "frobenius":
            self.norm_function = partial(torch.norm, p="fro")
        elif self.metric_config.norm_type == "nuclear":
            self.norm_function = partial(torch.norm, p="nuc")
        elif self.metric_config.norm_type == "inf":
            self.norm_function = partial(torch.norm, p=float("inf"))
        else:
            raise ValueError(f"Invalid norm_type: {self.metric_config.norm_type}")

    # NOTE: Any component is valid for the norm metric.
    def validate_component(self, component_config: BaseComponentConfig) -> None: ...

    def compute_metric(self, component_layer_data: torch.Tensor) -> float:
        """
        Computes the norm of the given component data.

        Args:
            component_layer_data: The component data to compute the norm of.

        Returns:
            The norm of the component data.
        """
        return self.norm_function(component_layer_data).item()
