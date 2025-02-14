"""
PER metrics compute the Proportional Effective Rank (PER) of activations or gradients.

The PER is defined as the  entropy over the normalised singular values of a given matrix.
Formally, if we let S = [s_1, ..., s_n] be the singular values of a parameter matrix P, then the PER is given by:

PER(P) = - sum(s_i / sum(s)) * log2(s_i / sum(s)) for i = 1 to n

where s = sum(s) is the sum of the singular values.
"""

from ._registry import register_metric
from .base import BaseMetric
from src.config.learning_dynamics import BaseMetricConfig

import torch


@register_metric("per")
class PERMetric(BaseMetric):
    """
    Base class for PER metrics.

    NOTE: This class provides
    """

    def __init__(self, metric_config: BaseMetricConfig, *args):
        super().__init__(metric_config, *args)

        if self.metric_config.data_type not in ["weights", "gradients"]:
            raise ValueError(f"Invalid data_type for: {self.metric_config.data_type}")

    def _compute_per(self, layer_data: torch.Tensor):
        """
        Computes the PER of a given layer.
        """

        layer_singular_values = torch.svd(layer_data).S

        # standardize singular values
        layer_singular_values = layer_singular_values / layer_singular_values.sum()

        # compute effective rank (ER) and proportional effective rank (PER)
        layer_er = torch.exp(
            -torch.sum(layer_singular_values * torch.log(layer_singular_values))
        ).item()
        layer_per = layer_er / len(layer_singular_values)

        return layer_per

    def compute(self, data: dict):
        """
        Computes the norm of the given data.
        """

        model_prefix = self._get_model_prefix(data["activations"])

        per_values = {}

        for component in self.metric_config.components:
            per_values[component.name] = {}

            for layer_idx in component.layers:
                # TODO -- Define special components

                assert isinstance(component.layer_suffixes, str)

                layer_name = f"{model_prefix}{layer_idx}.{component.layer_suffixes}"
                layer_data = data[self.metric_config.data_type][layer_name]
                layer_per = self._compute_per(layer_data)
                per_values[component.name][layer_name] = layer_per

        return per_values
