"""
Simple components are those that are a single layer. For example, the weight matrix of a layer is
a single component. In other words, simple components are those that can just be extracted directly
from the stored out checkpoint data without much additional computation.
"""

from src.components._base import BaseComponent
from src.components._registry import register_component
from src.utils.exceptions import InvalidComponentError

# typing imports
import torch
from typing import Dict, Any

from src.config._base import BaseComponentConfig


@register_component("simple")
class SimpleComponent(BaseComponent):
    """
    Simple component is a component that is a single layer. For example, the weight matrix of a layer is
    a single component.
    """

    def validate_component(self, component_config: BaseComponentConfig) -> None:
        """
        Check the component config; components should specify the required keys in the component
        config by overriding this method.
        """
        if component_config.data_type not in ["activations", "weights", "gradients"]:
            raise InvalidComponentError(
                f"Simple component only supports activations, weights, or gradients, not {component_config.data_type}."
            )

    def __call__(
        self,
        checkpoint_states: Dict[str, Dict[str, torch.Tensor]],
        component_config: BaseComponentConfig,
    ) -> Dict[str, Any]:
        """
        Given a dictionary of checkpoint data, extract the activations, weights, or gradients for
        the given layer suffix and layer.

        Args:
            checkpoint_states: Checkpoint data (activations, weights, gradients)
            component_config: The component configuration.

        Returns:
            A dictionary mapping layer names to MLP activations.
        """

        checkpoint_layer_component = {}

        _data = checkpoint_states[component_config.data_type]
        _model_prefix = self.get_model_prefix(checkpoint_states)

        for layer_idx in component_config.layers:
            layer_component = _data[
                f"{_model_prefix}{layer_idx}.{component_config.layer_suffixes}"
            ]
            checkpoint_layer_component[
                f"{_model_prefix}{layer_idx}.{component_config.layer_suffixes}.{component_config.data_type}"
            ] = layer_component

        return checkpoint_layer_component
