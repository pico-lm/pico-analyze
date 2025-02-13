"""
CKA (Centered Kernel Alignment) is a comparative metric for comparing how similar two
sets of activations are between two different checkpoints.
"""

from ._registry import register_metric
from .base import BaseComparativeMetric

import re
from lib import cka

# Typing imports
from src.config.learning_dynamics import BaseMetricConfig
import torch
from typing import Dict, List


@register_metric("cka")
class CKA(BaseComparativeMetric):
    """
    Class for computing CKA (Centered Kernel Alignment) between two activations.

    The CKA is computed between the OV circuit activations and the MLP activations. Formally,
    CKA(A, B) = (K(A, A)^T K(B, B)^T) / sqrt((K(A, A)^T K(A, A)^T) * (K(B, B)^T K(B, B)^T))

    where K(A, B) is the kernel matrix between the activations A and B.

    Reference: https://arxiv.org/pdf/1905.00414.pdf

    """

    def __init__(self, metric_config: BaseMetricConfig, run_config: dict) -> None:
        super().__init__(metric_config, run_config)

        # NOTE: these are all things that we should be able to get from the run config

        self.hidden_dim = self.run_config["model"].get("d_model", None)
        if self.hidden_dim is None:
            raise ValueError("d_model is not set in the run config")

        self.attention_n_heads = self.run_config["model"].get("attention_n_heads", None)
        if self.attention_n_heads is None:
            raise ValueError("attention_n_heads is not set in the run config")

        self.attention_head_dim = self.hidden_dim // self.attention_n_heads

        self.attention_n_kv_heads = self.run_config["model"].get(
            "attention_n_kv_heads", self.attention_n_heads
        )
        if self.attention_n_kv_heads is None:
            raise ValueError("attention_n_kv_heads is not set in the run config")

    def preprocess_data(self, data: dict):
        """
        Preprocesses the data to compute a metric.
        """

        checkpoint_activations = data["activations"]
        checkpoint_weights = data["weights"]

        preprocessed_data = {}
        for component in self.metric_config.components:
            if component["name"] == "ov_circuit":
                component_data = self.get_checkpoint_ov_activation(
                    checkpoint_activations,
                    checkpoint_weights,
                    component["layer_suffixes"],
                    component["layers"],
                )
            elif component["name"] == "mlp":
                component_data = self.get_mlp_activation(
                    checkpoint_activations,
                    component["layer_suffixes"],
                    component["layers"],
                )
            else:
                raise ValueError(f"CKA Metric: Unknown component: {component['name']}")

            preprocessed_data[component["name"]] = component_data

        return preprocessed_data

    def compute_comparison(
        self, source_data: dict, target_data: dict
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes the CKA between two sets of data.

        NOTE: The returned dictionary will have the same keys as the source (target) data dictionary.
        The values will simply be the CKA values between the source and target data.

        For example, if the source data dictionary contains data for the OV-Circuit at layer 0 and 1,
        the returned dictionary will have the following form:
        {
            "ov_circuit": {
                "layer_0": 0.5, # CKA between source OV-Circuit layer 0 and target OV-Circuit layer 0
                "layer_1": 0.3, # CKA between source OV-Circuit layer 1 and target OV-Circuit layer 1
            }
        }

        """
        cka_values = {}

        for src_component_name, src_component_data in source_data.items():
            cka_values[src_component_name] = {}
            for (
                src_component_layer_name,
                src_component_layer_data,
            ) in src_component_data.items():
                tgt_component_layer_data = target_data[src_component_name][
                    src_component_layer_name
                ]

                # NOTE: The CKA implementation expects float32 numpy darrays
                np_src_component_layer_data = src_component_layer_data.to(
                    dtype=torch.float32
                ).numpy()
                np_tgt_component_layer_data = tgt_component_layer_data.to(
                    dtype=torch.float32
                ).numpy()

                cka_value = cka.feature_space_linear_cka(
                    np_src_component_layer_data, np_tgt_component_layer_data
                )

                cka_values[src_component_name][src_component_layer_name] = cka_value

        return cka_values

    ####
    # Helper functions
    ####

    def _get_model_prefix(self, checkpoint_activation: Dict[str, torch.Tensor]):
        """
        Get the model prefix from the checkpoint activation keys.
        """

        return re.match(r"[^\d]+", list(checkpoint_activation.keys())[0]).group(0)

    """
    Helper functions for getting component states. What a 'component' is, can be a bit arbitrary. 
    The naming is meant to encapsulate the fact that a component can be the activations of the a 
    feed-forward layer, or those of an induction head in the nomenclature of mechanistic interpretability. 
   """

    def get_checkpoint_ov_activation(
        self,
        checkpoint_activation: Dict[str, torch.Tensor],
        checkpoint_weights: Dict[str, torch.Tensor],
        layer_suffixes: Dict[str, str],
        layers: List[int],
    ):
        """
        Compute the 'OV-Circuit' component. The idea of an OV-Circuit is to take the value activations
        and use the output projection of the model to compute the OV-Circuit activations.

        The OV-Circuit activations are computed as:
        OV(a, B) = a^T B

        where a is the value activation and B is the output projection of the model.

        The reason why the OV-Circuit is interesting is that in an attention module, the value
        and output projections always operate jointly, and write into the 'residual stream'.

        Args:
            checkpoint_activation: A dictionary mapping layer names to activations.
            checkpoint_weights: A dictionary mapping layer names to weights.
            layer_suffixes: A dictionary mapping layer names to suffixes.
            layers: A list of layers to compute the OV-Circuit activations for.

        Returns:
            A dictionary mapping layer names to OV-Circuit activations.
        """

        checkpoint_ov_activation = {}

        _model_prefix = self._get_model_prefix(checkpoint_activation)

        for layer_idx in layers:
            layer_value_activation = checkpoint_activation[
                f"{_model_prefix}{layer_idx}.{layer_suffixes['value_layer']}"
            ]
            layer_output_projection = checkpoint_weights[
                f"{_model_prefix}{layer_idx}.{layer_suffixes['output_layer']}"
            ]

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
                    layer_value_activation[
                        :, start_value_activation:end_value_activation
                    ]
                    @ layer_output_projection[
                        :,
                        head_idx * self.attention_head_dim : (head_idx + 1)
                        * self.attention_head_dim,
                    ].T
                )

                checkpoint_ov_activation[
                    f"{_model_prefix}{layer_idx}.ov_circuit.heads.{head_idx}"
                ] = ov_activation_per_head

        return checkpoint_ov_activation

    def get_mlp_activation(
        self,
        checkpoint_activation: Dict[str, torch.Tensor],
        layer_suffixes: Dict[str, str],
        layers: List[int],
    ):
        """
        Computing the MLP activations - component. There's not much to compute, instead we
        simply extract the activations from the checkpoint activations.

        Args:
            checkpoint_activation: A dictionary mapping layer names to activations.
            layer_suffixes: A dictionary mapping layer names to suffixes.
            layers: A list of layers to compute the MLP activations for.

        Returns:
            A dictionary mapping layer names to MLP activations.
        """

        checkpoint_mlp_activation = {}

        _model_prefix = self._get_model_prefix(checkpoint_activation)

        for layer_idx in layers:
            mlp_activation = checkpoint_activation[
                f"{_model_prefix}{layer_idx}.{layer_suffixes['mlp']}"
            ]
            checkpoint_mlp_activation[f"{_model_prefix}{layer_idx}.mlp"] = (
                mlp_activation
            )

        return checkpoint_mlp_activation
