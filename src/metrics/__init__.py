# ruff: noqa: F401

# NOTE: Importing these metrics automatically adds them to the METRIC_REGISTRY
from typing import Any, Dict

# Typing
from src.config.learning_dynamics import BaseMetricConfig

# Registry
from ._registry import METRIC_REGISTRY
from .base import BaseComparativeMetric, BaseMetric

# Comparative Metrics (between two checkpoints)
from .cka import CKAMetric
from .condition_number import ConditionNumberMetric
from .gini import GiniMetric
from .hoyer import HoyerMetric

# Implemented Metrics
# Base Metrics
from .norm import NormMetric
from .per import PERMetric
from .pwcca import PWCCAMetric


def get_metric(
    metric_config: BaseMetricConfig, training_config: Dict[str, Any]
) -> BaseMetric:
    """
    Loads a metric from the metrics directory.
    """
    return METRIC_REGISTRY[metric_config.metric_name](metric_config, training_config)
