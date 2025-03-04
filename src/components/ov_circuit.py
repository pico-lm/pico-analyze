"""
Compound component are those that are composed of multiple-layers; for example, the
OV circuit is a compound component that is composed of the OV circuit layers.
"""

from src.components._base import BaseComponent
from src.components._registry import register_component

from functools import lru_cache

# typing imports
import torch
from typing import Dict, Any
from src.config._base import BaseComponentConfig


@register_component("ov_circuit")
class OVComponent(BaseComponent):
    """
    Compute the 'OV-Circuit' component. The idea of an OV-Circuit stems from the observation
    that in an attention module, the value and the output projections always operate jointly,
    and write into the 'residual stream'. Thus, it makes sense to treat the Output and Value
    matrices as a single 'OV-Circuit' matrix, which we can compute the activations and weights for.

    To read more about the OV-Circuit, see:
        https://transformer-circuits.pub/2021/framework/index.html
    """

    def __init__(self, training_config: Dict[str, Any]):
        super().__init__(training_config)

        self.d_model = training_config["model"]["d_model"]

        self.attention_n_heads = training_config["model"]["attention_n_heads"]
        self.attention_n_kv_heads = training_config["model"]["attention_n_kv_heads"]
        self.attention_head_dim = self.d_model // self.attention_n_heads

    @lru_cache(maxsize=50)
    def compute_ov_activations(
        self,
        layer_value_activation: torch.Tensor,
        layer_output_projection: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the OV activations for a single layer. Uses a cache to speed up the computation,
        if the component is used across multiple metrics.

        Args:
            layer_value_activation: The value activations for the layer.
            layer_output_projection: The output projection for the layer.

        Returns:
            A dictionary mapping head indices to OV component activations.
        """
        layer_ov_activation = {}

        for head_idx in range(self.attention_n_heads):
            kv_head_idx = head_idx // (
                self.attention_n_heads // self.attention_n_kv_heads
            )

            if layer_value_activation.dtype != layer_output_projection.dtype:
                # NOTE: activations might be stored as memory efficient floats (e.g. bfloat16)
                # so we need to make sure we cast to the same type as the weights
                layer_value_activation = layer_value_activation.to(
                    layer_output_projection.dtype
                )

            start_value_activation = kv_head_idx * self.attention_head_dim
            end_value_activation = (kv_head_idx + 1) * self.attention_head_dim

            ov_activation_per_head = (
                layer_value_activation[:, start_value_activation:end_value_activation]
                @ layer_output_projection[
                    :,
                    head_idx * self.attention_head_dim : (head_idx + 1)
                    * self.attention_head_dim,
                ].T
            )

            layer_ov_activation[f"{head_idx}"] = ov_activation_per_head

        return layer_ov_activation

    @lru_cache(maxsize=50)
    def compute_ov_weights(
        self,
        layer_value_projection: torch.Tensor,
        layer_output_projection: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the OV weights for a single layer. Uses a cache to speed up the computation,
        if the component is used across multiple metrics.
        """

        layer_ov_weights = {}

        for head_idx in range(self.attention_n_heads):
            kv_head_idx = head_idx // (
                self.attention_n_heads // self.attention_n_kv_heads
            )

            start_value_projection = kv_head_idx * self.attention_head_dim
            end_value_projection = (kv_head_idx + 1) * self.attention_head_dim

            start_output_projection = head_idx * self.attention_head_dim
            end_output_projection = (head_idx + 1) * self.attention_head_dim

            layer_ov_weights[f"{head_idx}"] = (
                layer_value_projection[start_value_projection:end_value_projection, :]
                @ layer_output_projection[
                    :, start_output_projection:end_output_projection
                ]
            )

        return layer_ov_weights

    def check_component_config(self, component_config: BaseComponentConfig) -> None:
        """
        Check the component config; components should specify the required keys in the component
        config by overriding this method.
        """
        super().check_component_config(component_config)

        # NOTE: We only support activations and weights for the OV circuit component
        if component_config.data_type not in ["activations", "weights"]:
            raise ValueError(
                f"Invalid component data_type for OVComponent: {component_config.data_type}"
            )

        assert (
            "value_layer" in component_config.layer_suffixes
            and "output_layer" in component_config.layer_suffixes
        ), "Layer suffixes must contain value_layer and output_layer"

    def __call__(
        self,
        checkpoint_states: Dict[str, Dict[str, torch.Tensor]],
        component_config: BaseComponentConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        Generates the OV circuit component. The OV circuit component is a compound component
        that is composed of the value and output projections of the model.

        Args:
            checkpoint_states: Dict[str, Dict[str, torch.Tensor]] -- the checkpoint states
                to generate the component for.
            component_config: BaseComponentConfig -- the component configuration.

        Returns:
            Dict[str, torch.Tensor] -- the OV circuit component; mapping layer names to OV circuit
                activations.
        """

        super().__call__(checkpoint_states, component_config)

        layer_suffixes = component_config.layer_suffixes

        checkpoint_activation = checkpoint_states[component_config.data_type]
        checkpoint_weights = checkpoint_states["weights"]

        checkpoint_layer_component = {}

        _model_prefix = self.get_model_prefix(checkpoint_states)

        for layer_idx in component_config.layers:
            if component_config.data_type == "activations":
                layer_value_activation = checkpoint_activation[
                    f"{_model_prefix}{layer_idx}.{layer_suffixes['value_layer']}"
                ]
            elif component_config.data_type == "weights":
                layer_value_projection = checkpoint_weights[
                    f"{_model_prefix}{layer_idx}.{layer_suffixes['value_layer']}"
                ]

            layer_output_projection = checkpoint_weights[
                f"{_model_prefix}{layer_idx}.{layer_suffixes['output_layer']}"
            ]

            if component_config.data_type == "activations":
                ov_component = self.compute_ov_activations(
                    layer_value_activation, layer_output_projection
                )
            elif component_config.data_type == "weights":
                ov_component = self.compute_ov_weights(
                    layer_value_projection, layer_output_projection
                )

            for head_idx, ov_component_head in ov_component.items():
                checkpoint_layer_component[
                    f"{_model_prefix}{layer_idx}.ov_circuit.{component_config.data_type}.heads.{head_idx}"
                ] = ov_component_head

        return checkpoint_layer_component
