# ruff: noqa: F401

# NOTE: Importing these components automatically adds them to the COMPONENT_REGISTRY
from typing import Any, Dict

from src.components._registry import COMPONENT_REGISTRY
from src.components.base import BaseComponent
from src.components.ov_circuit import OVComponent
from src.components.simple import SimpleComponent

# typing imports
from src.config.base import BaseComponentConfig

COMPONENT_CACHE = {}


def get_component(
    component_config: BaseComponentConfig, run_config: Dict[str, Any]
) -> BaseComponent:
    """
    Get a component from the registry. We cache components to avoid re-initializing them,
    if the same component is used across multiple metrics. One benefit of doing so, is that
    components can maintain an internal state to avoid re-computing the same component multiple
    times on the same layer.

    Args:
        component_config: BaseComponentConfig -- the component configuration.
        run_config: Dict[str, Any] -- the run config.

    Returns:
        BaseComponent -- the component.
    """

    component_name = component_config.component_name
    # track if the component name and type
    if component_name not in COMPONENT_CACHE:
        COMPONENT_CACHE[component_name] = COMPONENT_REGISTRY[component_name](run_config)

    return COMPONENT_CACHE[component_name]
