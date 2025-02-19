"""
Configurations for metrics. Recall there are two types of metrics:

1. BaseMetricConfig: these are metrics that are computed on a single checkpoint; e.g. the
    norm of a layer at a given checkpoint.
2. BaseComparativeMetricConfig: these are metrics that are computed on a pair of checkpoints; e.g.
    the CKA between two layers at two different checkpoints to measure how similar the two layers
    are between the two checkpoints.
"""

from dataclasses import dataclass

from src.config.base import BaseMetricConfig, BaseComparativeMetricConfig
from src.config._registry import register_metric_config


# -----------------
# Single Checkpoint (Base) Metrics
# -----------------


@dataclass
@register_metric_config("norm")
class NormConfig(BaseMetricConfig):
    """
    Configuration for a norm metric.
    """

    # NOTE: used to specify what type of norm to compute:
    #       options are "Frobenius", "spectral", "max"
    norm_type: str = None


@dataclass
@register_metric_config("per")
class PERConfig(BaseMetricConfig):
    """
    Configuration for the Proportional Effective Rank (PER) metric.
    The PER is a metric that measures the effective rank of a matrix, and is defined in:
        Tending Towards Stability: Convergence Challenges in Small Language Models
        https://aclanthology.org/2024.findings-emnlp.187/
    """

    ...


# -----------------
# Multi-Checkpoint (Comparative) Metrics
# -----------------


@dataclass
@register_metric_config("cka")
class CKAConfig(BaseComparativeMetricConfig):
    """
    Configuration for the CKA metric; a comparative metric that computes the similarity between two
    layers' activations at two different checkpoints.
    """

    ...
