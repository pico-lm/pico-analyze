from typing import Mapping, Type, TypeVar

from src.components.base import BaseComponent

T = TypeVar("T", bound=BaseComponent)

COMPONENT_REGISTRY: Mapping[str, Type[T]] = {}


def register_component(name: str):
    """
    Decorator to register a component class with the ComponentRegistry.
    """

    def _register(cls: Type[T]) -> Type[T]:
        COMPONENT_REGISTRY[name] = cls
        return cls

    return _register
