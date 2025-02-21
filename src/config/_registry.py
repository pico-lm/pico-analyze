from typing import Type, TypeVar

from src.config._base import BaseMetricConfig

T = TypeVar("T", bound=BaseMetricConfig)

METRIC_CONFIG_REGISTRY: dict[str, Type[T]] = {}


def register_metric_config(metric_name: str):
    """
    Decorator to register a metric config class with the MetricConfigRegistry.
    """

    def decorator(cls):
        METRIC_CONFIG_REGISTRY[metric_name] = cls
        return cls

    return decorator
