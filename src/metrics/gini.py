"""
Gini coefficient is a measure of the 'inequality' of a distribution; we use it indirectly to
measure the sparsity of a matrix.

"""

import torch

from src.config._base import BaseComponentConfig
from src.metrics._registry import register_metric
from src.metrics.base import BaseMetric


@register_metric("gini")
class GiniMetric(BaseMetric):
    """
    Compute the Gini coefficient of some component data which is a rough approximation of the
    sparsity of a matrix.

    Formally, if we let x = [x_1, ..., x_n] be the data, then the Gini coefficient is given by:

        G(x) = 1 - sum(x_i) / sum(x) for i = 1 to n

    where x_i is the i-th element of the data, and x is the sum of all the elements in the data.
    """

    # NOTE: Any component is valid for the Gini metric.
    def validate_component(self, component_config: BaseComponentConfig) -> None: ...

    def compute_metric(self, component_layer_data: torch.Tensor) -> float:
        """
        Compute the Gini coefficient of some component data.

        The Gini coefficient measures inequality in a distribution, with values ranging from 0
        (perfect equality) to 1 (perfect inequality).

        This implementation uses a more memory-efficient algorithm that avoids creating
        the full pairwise difference matrix.

        Args:
            component_layer_data: Tensor containing the data to analyze

        Returns:
            float: The computed Gini coefficient
        """
        # Reshape the input tensor to a 1D array
        x = component_layer_data.flatten()
        x = torch.abs(x)

        # Sort the flattened vector in ascending order
        x_sorted, _ = torch.sort(x)
        n = x_sorted.shape[0]
        if n == 0:
            return 0.0  # Edge case if the matrix is empty

        # Compute the mean denominator
        total = x_sorted.sum()
        if total == 0:
            return 0.0  # If all entries are zero, Gini is 0 by convention

        # Apply the formula
        # sum_{i=1 to n} of (2i - n - 1) * x_sorted[i-1]
        idx = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
        numerator = ((2 * idx - n - 1) * x_sorted).sum()

        G = numerator / (n * total)
        return G.item()
