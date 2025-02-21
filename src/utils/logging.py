"""
Miscellaneous logging utilities.
"""

from rich.console import Console
from rich.panel import Panel
from io import StringIO
import yaml


def pretty_print_config(logger, title: str, config: dict) -> None:
    """Pretty print config with rich formatting."""
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
