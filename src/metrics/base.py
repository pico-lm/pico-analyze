"""
Base class for all metrics.
"""

from abc import ABC, abstractmethod
from src.components import get_component

# Typing
import torch
from src.config.learning_dynamics import BaseMetricConfig
from typing import Dict, Any, List


class BaseMetric(ABC):
    """
    Base class for all metrics.
    """

    def __init__(self, metric_config: BaseMetricConfig, run_config: Dict[str, Any]):
        """
        Initialize the metric with the given metric config and a run config (the config used during
        training that specifies the model architecture, etc.).

        To see an example of a run_config, see the training_config.yaml file in the demo run of Pico:
            https://huggingface.co/pico-lm/demo/blob/demo-1/training_config.yaml

        The run_config is used to setup and compute the components that metrics are computed on.

        Args:
            metric_config: BaseMetricConfig -- the metric config to use for the metric.
            run_config: Dict[str, Any] -- the run config to use for the metric.
        """
        self.metric_config = metric_config

        # Setup components
        self.components = []
        for component_config in self.metric_config.components:
            component = get_component(component_config, run_config)
            self.components.append(component)

    def compute_components(
        self, checkpoint_states: Dict[str, Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Preprocesses the checkpoint states to generate the components. This is a helper function that
        is used by the compute method.

        Args:
            checkpoint_states: Dict[str, Dict[str, torch.Tensor]] -- the initial checkpoint states
                to preprocess and generate the components.
            **kwargs: Any -- additional arguments to pass to the component setup.

        Returns:
            components_data_list: List[Dict[str, torch.Tensor]] -- a list of dictionaries mapping component
                names to component data; each component data is a dictionary mapping layer names to tensors
                that are the preprocessed components at that layer.
        """
        component_data_list = []
        for component, component_config in zip(
            self.components, self.metric_config.components
        ):
            component_data_list.append(component(checkpoint_states, component_config))

        return component_data_list

    @abstractmethod
    def compute_metric(self, component_layer_data: torch.Tensor) -> float:
        """
        Computes the desired metric on a single component at a given layer.

        Args:
            component_layer_data: torch.Tensor -- the data for a component at a given layer.

        Returns:
            torch.Tensor -- the computed metric for the given component at the given layer.
        """
        pass

    def __call__(
        self, checkpoint_states: Dict[str, Dict[str, torch.Tensor]]
    ) -> List[Dict[str, float]]:
        """
        Computed the desired metrics on the specified components in the metric config. Reads in the
        original data, preprocesses it into components, and then computes the desired metrics on each
        component.

        Args:
            checkpoint_states: Dict[str, Any] -- the checkpoint states to compute the metric on.

        Returns:
            component_metrics_list: List[Dict[str, float]] -- a list of dictionaries mapping component names
                across layers to computed metrics.

                NOTE: The order in which the components are returned is the same as the order in
                which they are specified in the metric config.
        """

        component_data_list = self.compute_components(checkpoint_states)

        component_metrics_list = []

        for component_data in component_data_list:
            # component_data will be a dictionary mapping layer names to tensors
            component_metric = {}

            for _component_layer_name, _component_layer_data in component_data.items():
                component_metric[_component_layer_name] = self.compute_metric(
                    _component_layer_data
                )

            component_metrics_list.append(component_metric)

        return component_metrics_list


class BaseComparativeMetric(BaseMetric):
    """
    Base class for comparative metrics.

    The idea of comparative metrics is that these metrics compare the current checkpoint to a
    given target checkpoint. For example, we can compute the CKA between the current checkpoint and
    the target checkpoint.

    NOTE: the target data (e.g. precomputed target components) are set using the set_target method.
    """

    def __init__(self, metric_config: BaseMetricConfig, run_config: Dict[str, Any]):
        """
        Initialize the comparative metric as a base metric, but with a target checkpoint.
        """
        super().__init__(metric_config, run_config)

        self._target_component_data_list = None

    def set_target(self, checkpoint_states: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """
        Set the target checkpoint data which is stored as the already preprocessed components.
        We set the target data before computing metrics so that we can compute the metric between
        the source and target checkpoints.

        Args:
            checkpoint_states: Dict[str, Dict[str, torch.Tensor]] -- the target checkpoint states.
        """
        self._target_component_data_list = self.compute_components(checkpoint_states)

    @abstractmethod
    def compute_metric(
        self,
        source_component_layer_data: torch.Tensor,
        target_component_layer_data: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Computes the given metric between two components. Unlike the BaseMetric class, this method
        takes in two tensors (source and target) and computes the metric between them.

        Args:
            source_component_layer_data: torch.Tensor -- the source component data at a given layer.
            target_component_layer_data: torch.Tensor -- the target component data at a given layer.

        Returns:
            float -- the computed metric.
        """
        raise NotImplementedError

    def __call__(
        self, source_checkpoint_states: Dict[str, Dict[str, torch.Tensor]]
    ) -> List[Dict[str, float]]:
        """
        Computes the metric between two sets of checkpoint states.

        NOTE: Can only be called if the target data is set; that is, that set_target() has been called
        with the checkpoint states of the target checkpoint.

        Args:
            source_checkpoint_states: Dict[str, Dict[str, torch.Tensor]] -- the source checkpoint
                states.

        Returns:
            component_metrics_list: List[Dict[str, float]] -- a list of dictionaries mapping components
                across layers to computed metrics.

                NOTE: The order in which the components are returned is the same as the order in
                which they are specified in the metric config.
        """
        if self._target_component_data_list is None:
            raise ValueError("Target data is not set. Call .set_target() first.")

        src_component_data_list = self.compute_components(source_checkpoint_states)

        component_metrics_list = []

        for src_component_data, target_component_data in zip(
            src_component_data_list, self._target_component_data_list
        ):
            # source_component_data will be a dictionary mapping layer names to tensors
            component_metric = {}

            for (
                component_layer_name,
                source_component_layer_data,
            ) in src_component_data.items():
                target_component_layer_data = target_component_data[
                    component_layer_name
                ]

                component_metric[component_layer_name] = self.compute_metric(
                    source_component_layer_data,
                    target_component_layer_data,
                )

            component_metrics_list.append(component_metric)

        return component_metrics_list
