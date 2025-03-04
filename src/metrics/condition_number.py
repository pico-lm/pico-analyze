"""
This module contains the implementation of the singular values metric.
"""

from src.metrics._registry import register_metric
from src.metrics.base import BaseMetric

import torch


@register_metric("condition_number")
class ConditionNumberMetric(BaseMetric):
    """
    This metric computes the condition number of some component data. The condition number is the
    ratio of the largest to smallest singular value of the input. It gives a measure of how
    sensitive the output is to small changes in the input.
    """

    def compute_metric(self, component_layer_data: torch.Tensor) -> float:
        """
        Computes the condition number of the given input.
        """

        # Compute the singular values of the input
        singular_values = torch.svd(component_layer_data).S

        # Compute the condition number
        condition_number = torch.max(singular_values) / torch.min(singular_values)

        return condition_number.item()
