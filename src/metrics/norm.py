"""
Norm metrics comptue the norm of weights, activations, gradients, etc.
"""

from ._registry import register_metric
from .base import BaseMetric
from src.config.learning_dynamics import BaseMetricConfig

import torch

from functools import partial


@register_metric("norm")
class NormMetric(BaseMetric):
    """
    Base class for norm metrics.

    NOTE: This class provides
    """

    def __init__(self, metric_config: BaseMetricConfig, *args):
        super().__init__(metric_config, *args)

        if self.metric_config.data_type not in ["weights", "activations", "gradients"]:
            raise ValueError(f"Invalid data_type: {self.metric_config.data_type}")

        # NOTE: We use the torch.norm function to compute the norm of the data.
        if self.metric_config.norm_type == "frobenius":
            self.norm_function = partial(torch.norm, p="fro")
        elif self.metric_config.norm_type == "nuclear":
            self.norm_function = partial(torch.norm, p="nuc")
        elif self.metric_config.norm_type == "inf":
            self.norm_function = partial(torch.norm, p=float("inf"))
        else:
            raise ValueError(f"Invalid norm_type: {self.metric_config.norm_type}")

    def compute(self, data: dict):
        """
        Computes the norm of the given data.
        """

        model_prefix = self._get_model_prefix(data["activations"])

        norm_values = {}

        for component in self.metric_config.components:
            norm_values[component.name] = {}

            for layer_idx in component.layers:
                # TODO -- Define special components
                # NOTE: we probably want to combine the computations for special components
                #       into a single function call.

                assert isinstance(component.layer_suffixes, str)

                layer_name = f"{model_prefix}{layer_idx}.{component.layer_suffixes}"
                layer_data = data[self.metric_config.data_type][layer_name]
                norm_values[component.name][layer_name] = self.norm_function(
                    layer_data
                ).item()

        return norm_values
