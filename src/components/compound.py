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
    Compute the 'OV-Circuit' component. The idea of an OV-Circuit is to take the value activations
    and use the output projection of the model to compute the OV-Circuit activations.

    The OV-Circuit activations are computed as:
        OV(a, B) = a^T B

    where a is the value activation and B is the output projection of the model.

    The reason why the OV-Circuit is interesting is that in an attention module, the value
    and output projections always operate jointly, and write into the 'residual stream'.
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
    def compute_layer_component(
        self,
        layer_value_activation: torch.Tensor,
        layer_output_projection: torch.Tensor,
        head_idx: int,
    ) -> torch.Tensor:
        """
        Compute the OV component for a single layer. Uses a cache to speed up the computation,
        if the component is used across multiple metrics.

        Args:
            layer_value_activation: The value activations for the layer.
            layer_output_projection: The output projection for the layer.
            head_idx: The index of the head to compute the OV component for.

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

            layer_ov_activation[f"head_{head_idx}"] = ov_activation_per_head

        return layer_ov_activation

    def __call__(
        self,
        checkpoint_states: Dict[str, Dict[str, torch.Tensor]],
        component_config: BaseComponentConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        Generates the OV circuit component.

        Args:
            checkpoint_states: Dict[str, Dict[str, torch.Tensor]] -- the checkpoint states
                to generate the component for.
            component_config: BaseComponentConfig -- the component configuration.

        Returns:
            Dict[str, torch.Tensor] -- the OV circuit component; mapping layer names to OV circuit
                activations.
        """

        layer_suffixes = component_config.layer_suffixes
        assert (
            "value_layer" in layer_suffixes and "output_layer" in layer_suffixes
        ), "Layer suffixes must contain value_layer and output_layer"

        checkpoint_activation = checkpoint_states["activations"]
        checkpoint_weights = checkpoint_states["weights"]

        checkpoint_ov_activation = {}

        _model_prefix = self.get_model_prefix(checkpoint_states)

        for layer_idx in component_config.layers:
            layer_value_activation = checkpoint_activation[
                f"{_model_prefix}{layer_idx}.{layer_suffixes['value_layer']}"
            ]
            layer_output_projection = checkpoint_weights[
                f"{_model_prefix}{layer_idx}.{layer_suffixes['output_layer']}"
            ]

            layer_ov_activation = self.compute_layer_component(
                layer_value_activation, layer_output_projection, layer_idx
            )

            for head_idx, ov_activation_head in layer_ov_activation.items():
                checkpoint_ov_activation[
                    f"{_model_prefix}{layer_idx}.ov_circuit.heads.{head_idx}"
                ] = ov_activation_head

        return checkpoint_ov_activation
