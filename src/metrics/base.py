"""
Base class for all metrics.
"""

from abc import ABC, abstractmethod
import re

# Typing
from src.config.learning_dynamics import BaseMetricConfig
from typing import Dict, Any
import torch


class BaseMetric(ABC):
    """
    Base class for all metrics.
    """

    def __init__(self, metric_config: BaseMetricConfig, run_config: Dict[str, Any]):
        """
        Initialize the metric with the given metric config and a run config (the config used during
        training that specifies the model architecture, etc.).
        """
        self.metric_config = metric_config
        self.run_config = run_config

    def _get_model_prefix(self, checkpoint_activation: Dict[str, torch.Tensor]):
        """
        Common helper function to get the model prefix from the checkpoint activation keys.

        The model prefix is the part of the key that is common to all the keys in the checkpoint
        activation dictionary. For example, if the checkpoint activation keys are:

        ```
        {
            "model.layer1.0.weight": torch.Tensor,
            "model.layer1.1.weight": torch.Tensor,
            "model.layer2.0.weight": torch.Tensor,
        }
        ```

        The model prefix is "model".
        """

        return re.match(r"[^\d]+", list(checkpoint_activation.keys())[0]).group(0)

    @abstractmethod
    def compute(self, data: Dict[str, Any]):
        """
        Computes the metric.
        """
        pass


class BaseComparativeMetric(BaseMetric):
    """
    Base class for comparative metrics.

    The idea of comparative metrics is that these metrics compare the current checkpoint to a
    given target checkpoint. For example, we can compute the CKA between the current checkpoint and
    the target checkpoint.

    NOTE: the target data is set using the set_target method.
    """

    def __init__(self, metric_config: BaseMetricConfig, run_config: Dict[str, Any]):
        """
        Initialize the comparative metric as a base metric, but with a target checkpoint.
        """
        super().__init__(metric_config, run_config)

        self._target_data = None

    def has_target(self) -> bool:
        """Check if target data is set"""
        return self._target_data is not None

    def set_target(self, data: Dict[str, Any]) -> None:
        """Set the target data"""
        self._target_data = self.preprocess_data(data)

    @abstractmethod
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocesses the data to compute a metric.
        """
        pass

    @abstractmethod
    def compute_comparison(
        self, source_data: Dict[str, Any], target_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Computes the given metric between two sets of data. Typically source_data will be the current
        checkpoint data, and target_data will be the target checkpoint data. If target_data is not
        provided, the metric should be computed between the source_data and the target data that was
        set using the set_target method.
        """
        raise NotImplementedError

    def compute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes the metric between two sets of data. Can only be called if the ready flag is set to True.
        """
        if not self.has_target:
            raise ValueError(
                "Target data is not set. Call compute_target_metric first."
            )

        self.compute_comparison(self.preprocess_data(data), self._target_data)
