from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WandbConfig:
    """
    Configuration for the Wandb experiment tracker.
    """

    entity: str = None
    project: str = None


@dataclass
class PicoReportConfig:
    """
    Configuration for Pico Report integration.

    Note: Requires PICO_API_KEY and PICO_LAB_HASH environment variables to be set.
    Optional: PICO_BASE_URL (defaults to https://picolabs.space/api/report)
    """

    lab_hash: Optional[str] = None

    # Git tracking: automatically create git commits for each analysis run
    # This captures the exact code state and links it to your analysis in the dashboard
    auto_commit: bool = True


@dataclass
class MonitoringConfig:
    """
    Configuration for the monitoring/logging of learning dynamics metrics.
    """

    output_dir: str = "analysis_results"

    save_to_wandb: bool = False
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Pico Labs - A platform to easily run and share experiments on the web
    # Automatically tracks analysis metrics to your private dashboard at https://picolabs.space
    save_to_picolabs: bool = False
    pico_report: PicoReportConfig = field(default_factory=PicoReportConfig)

    def __post_init__(self):
        """
        Post-initialization method to convert metric dictionaries to proper config objects. Used
        for loading in metrics from a yaml file where the metrics are specified as dictionaries.
        """
        if isinstance(self.wandb, dict):
            self.wandb = WandbConfig(**self.wandb)

        if isinstance(self.pico_report, dict):
            self.pico_report = PicoReportConfig(**self.pico_report)
