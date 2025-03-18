"""
Miscellaneous logging utilities.
"""

import logging
from io import StringIO

import yaml
from rich.console import Console
from rich.panel import Panel


def pretty_print_config(logger: logging.Logger, title: str, config: dict) -> None:
    """
    Pretty print config with rich formatting. We use the rich library to create a panel
    with a blue border and a title. We then convert the config to a YAML string and print
    it to the buffer.

    Args:
        logger: The logger instance
        title: The title of the config
        config: The config to print
    """
    # Create string buffer
    output = StringIO()
    console = Console(file=output, force_terminal=False)

    # Convert to YAML string first
    yaml_str = yaml.dump(
        config, default_flow_style=False, sort_keys=False, Dumper=yaml.SafeDumper
    )

    # Create formatted panel
    panel = Panel(
        yaml_str,
        title=f"[bold blue]{title}[/bold blue]",
        border_style="blue",
        padding=(0, 1),  # Reduced padding
        expand=False,  # Don't expand to terminal width
    )

    # Print to buffer
    console.print(panel)

    # Log the formatted output
    for line in output.getvalue().splitlines():
        logger.info(line)


def pretty_print_component_metrics(
    logger: logging.Logger, step: int, step_metrics: dict
):
    """
    Log multiple component metrics in a grouped and aligned format. We group the components by
    layer and sort them alphabetically. We also sort the metrics alphabetically. Finally,
    for each metric, we display the components in a simple bar chart.

    Args:
        logger: The logger instance
        step: The training step
        step_metrics: Dictionary of dictionaries, where each key is a metric name and
                      each value is a dictionary of component metrics
    """
    # Create a header for all metrics
    header = f"📊 Component Metrics at Step {step}"
    separator = "=" * len(header)

    logger.info(separator)
    logger.info(header)
    logger.info(separator)

    # Sort metrics alphabetically for consistent display
    sorted_metric_names = sorted(step_metrics.keys())

    for metric_name in sorted_metric_names:
        component_metrics_dict = step_metrics[metric_name]

        # Create a subheader for each metric
        metric_header = f"🔍 {metric_name.upper()} Metric"
        metric_separator = "-" * len(metric_header)

        logger.info(f"{metric_separator}")
        logger.info(metric_header)
        logger.info(metric_separator)

        # Group components by layer
        layer_groups = {}
        other_components = {}

        for component_key, metric_value in component_metrics_dict.items():
            # Extract layer number if present
            parts = component_key.split(".")
            layer_match = None
            layer_idx = None

            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        layer_match = f"layers.{layer_idx}"
                        break
                    except ValueError:
                        pass

            if layer_match:
                # Component belongs to a specific layer
                layer_name = f"Layer {layer_idx}"
                if layer_name not in layer_groups:
                    layer_groups[layer_name] = {
                        "_layer_idx": layer_idx,
                        "components": {},
                    }

                layer_groups[layer_name]["components"][component_key] = metric_value
            else:
                # Component doesn't belong to a specific layer
                # NOTE: this really shouldn't happen, but is a fallback for any components
                # that don't have a layer index.
                other_components[component_key] = metric_value

        # Sort layers by index
        sorted_layer_names = sorted(
            layer_groups.keys(), key=lambda x: layer_groups[x]["_layer_idx"]
        )

        # Display components by layer
        for layer_name in sorted_layer_names:
            logger.info(f"📌 {layer_name}")

            # Get components for this layer and sort them
            components = layer_groups[layer_name]["components"]

            # Sort components by name, but ensure head_0 comes before head_10
            def sort_key(name):
                parts = name.split(".")
                result = []
                for part in parts:
                    try:
                        result.append((0, int(part)))
                    except ValueError:
                        result.append((1, part))
                return result

            sorted_component_names = sorted(components.keys(), key=sort_key)

            # Find max length for alignment
            max_name_length = (
                max(len(name) for name in sorted_component_names)
                if sorted_component_names
                else 0
            )

            # Display each component
            for component_name in sorted_component_names:
                value = components[component_name]
                bar_length = min(round(value * 20), 20)  # Use round() instead of int()
                bar = "█" * bar_length + "░" * (20 - bar_length)
                logger.info(
                    f"  {component_name.ljust(max_name_length)} │ {bar} │ {value:.4f}"
                )

        # Display other components
        if other_components:
            logger.info("📌 Other Components")
            sorted_other_names = sorted(other_components.keys())
            max_other_length = (
                max(len(name) for name in sorted_other_names)
                if sorted_other_names
                else 0
            )

            for component_name in sorted_other_names:
                value = other_components[component_name]
                bar_length = min(round(value * 20), 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                logger.info(
                    f"  {component_name.ljust(max_other_length)} │ {bar} │ {value:.4f}"
                )
