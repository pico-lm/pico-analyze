"""
Gini coefficient is a measure of the 'inequality' of a distribution; we use it indirectly to
measure the sparsity of a matrix.

"""

from src.metrics._registry import register_metric
from src.metrics.base import BaseMetric
from src.config.learning_dynamics import BaseMetricConfig

import torch


@register_metric("gini")
class GiniMetric(BaseMetric):
    """
    Compute the Gini coefficient of some component data which is a rough approximation of the
    sparsity of a matrix.

    Formally, if we let x = [x_1, ..., x_n] be the data, then the Gini coefficient is given by:

        G(x) = 1 - sum(x_i) / sum(x) for i = 1 to n

    where x_i is the i-th element of the data, and x is the sum of all the elements in the data.
    """

    def __init__(self, metric_config: BaseMetricConfig, *args):
        super().__init__(metric_config, *args)

    def compute_metric(self, component_layer_data: torch.Tensor) -> float:
        """
        Compute the Gini coefficient of some component data.
        """

        # Reshape the input tensor to a 1D array
        x = component_layer_data.view(-1)
        n = x.shape[0]

        if n == 0:
            return 0.0

        mu = x.mean()
        if mu == 0:
            return 0.0

        # Expand to compare each element pair
        x_expanded = x.unsqueeze(0)  # shape (1, n)
        diff_matrix = torch.abs(x_expanded - x_expanded.T)  # shape (n, n)

        gini = diff_matrix.sum() / (2 * n**2 * mu)

        return gini.item()
