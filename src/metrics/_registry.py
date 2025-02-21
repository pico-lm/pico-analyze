from typing import Mapping, Type, TypeVar

from src.metrics.base import BaseMetric

T = TypeVar("T", bound=BaseMetric)

METRIC_REGISTRY: Mapping[str, Type[T]] = {}


def register_metric(name: str):
    """
    Decorator to register a metric class with the MetricRegistry.
    """

    def _register(cls: Type[T]) -> Type[T]:
        METRIC_REGISTRY[name] = cls
        return cls

    return _register
