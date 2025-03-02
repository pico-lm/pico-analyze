"""
Hoyer's sparsity metric is a measure of the sparsity of a matrix.
"""

from src.metrics._registry import register_metric
from src.metrics.base import BaseMetric
from src.config.learning_dynamics import BaseMetricConfig

import torch


@register_metric("hoyer")
class HoyerMetric(BaseMetric):
    """
    Hoyer's sparsity metric is a measure of the sparsity of a matrix. Formally, it is defined as:

        Hoyer(P) = (||P||_0 - 1) / (||P||_0 * n)

    where P is the parameter matrix, ||.||_0 is the number of non-zero elements in a matrix, and n is the
    total number of elements in the matrix.
    """

    def __init__(self, metric_config: BaseMetricConfig, *args):
        super().__init__(metric_config, *args)

    def compute_metric(self, component_layer_data: torch.Tensor) -> float:
        """
        Computes the Hoyer sparsity metric for a given component layer data.
        """

        # Compute the number of non-zero elements in the matrix
        num_non_zero = (component_layer_data != 0).sum().item()

        # Compute the total number of elements in the matrix
        total_elements = component_layer_data.numel()

        # Compute the Hoyer sparsity metric
        hoyer_sparsity = (num_non_zero - 1) / (num_non_zero * total_elements)

        return hoyer_sparsity
