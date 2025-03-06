"""
CKA (Centered Kernel Alignment) is a comparative metric for comparing how similar two
sets of activations are between two different checkpoints.
"""

from src.metrics._registry import register_metric
from src.metrics.base import BaseComparativeMetric
from src.config._base import BaseComponentConfig
from lib import cka

# Typing imports
import torch


@register_metric("cka")
class CKAMetric(BaseComparativeMetric):
    """
    Class for computing CKA (Centered Kernel Alignment) between two activations.

    The CKA is computed between the OV circuit activations and the MLP activations. Formally,
    CKA(A, B) = (K(A, A)^T K(B, B)^T) / sqrt((K(A, A)^T K(A, A)^T) * (K(B, B)^T K(B, B)^T))

    where K(A, B) is the kernel matrix between the activations A and B.

    Reference: https://arxiv.org/pdf/1905.00414.pdf

    """

    def valid_component_config(self, component_config: BaseComponentConfig) -> bool:
        """
        The CKA metric is only valid for activations.
        """
        if component_config.data_type not in ["activations"]:
            return False

        return True

    def compute_metric(
        self,
        source_component_layer_data: torch.Tensor,
        target_component_layer_data: torch.Tensor,
    ) -> float:
        """
        Computes the CKA between two sets of source and target component layer activations.
        """

        # NOTE: The CKA implementation expects float32 numpy darrays
        np_src_component_layer_data = source_component_layer_data.to(
            dtype=torch.float32
        ).numpy()
        np_tgt_component_layer_data = target_component_layer_data.to(
            dtype=torch.float32
        ).numpy()

        cka_value = cka.feature_space_linear_cka(
            np_src_component_layer_data, np_tgt_component_layer_data
        )

        return cka_value
