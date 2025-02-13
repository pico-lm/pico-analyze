from typing import Mapping, Type, TypeVar

from .base import BaseMetric

T = TypeVar("T", bound=BaseMetric)

METRIC_REGISTRY: Mapping[str, Type[BaseMetric]] = {}


def register_metric(name: str):
    def _register(cls: Type[T]) -> Type[T]:
        METRIC_REGISTRY[name] = cls
        return cls

    return _register
