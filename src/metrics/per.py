"""
PER metric computes the Proportional Effective Rank (PER) of activations or gradients.
"""

from src.metrics._registry import register_metric
from src.metrics.base import BaseMetric
from src.config.learning_dynamics import BaseMetricConfig
from src.config._base import BaseComponentConfig

import torch


@register_metric("per")
class PERMetric(BaseMetric):
    """
    Compute the Proportional Effective Rank (PER) of some component data. The PER is defined as the
    entropy over the normalised singular values of a given matrix.

    Formally, if we let S = [s_1, ..., s_n] be the singular values of a parameter matrix P, then the PER is given by:

        PER(P) = - sum(s_i / sum(s)) * log2(s_i / sum(s)) for i = 1 to n

    where s = sum(s) is the sum of the singular values.
    """

    def __init__(self, metric_config: BaseMetricConfig, *args):
        super().__init__(metric_config, *args)

        for component in self.metric_config.components:
            if component.data_type not in ["weights", "gradients"]:
                raise ValueError(
                    f"Invalid component data_type for PERMetric: {component.data_type}"
                )

    def valid_component_config(self, component_config: BaseComponentConfig) -> bool:
        """
        The PER metric is only valid for weights and gradients.
        """
        if component_config.data_type not in ["weights", "gradients"]:
            return False

        return True

    def compute_metric(self, component_layer_data: torch.Tensor) -> float:
        """
        Computes the PER of a given layer.
        """

        layer_singular_values = torch.svd(component_layer_data).S

        # standardize singular values
        layer_singular_values = layer_singular_values / layer_singular_values.sum()

        # compute effective rank (ER) and proportional effective rank (PER)
        layer_er = torch.exp(
            -torch.sum(layer_singular_values * torch.log(layer_singular_values))
        ).item()
        layer_per = layer_er / len(layer_singular_values)

        return layer_per
