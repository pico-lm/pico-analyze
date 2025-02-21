from dataclasses import dataclass, field


@dataclass
class WandbConfig:
    """
    Configuration for the Wandb experiment tracker.
    """

    entity: str = None
    project: str = None


@dataclass
class MonitoringConfig:
    """
    Configuration for the monitoring/logging of learning dynamics metrics.
    """

    output_dir: str = "analysis_results"

    save_to_wandb: bool = False
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        """
        Post-initialization method to convert metric dictionaries to proper config objects. Used
        for loading in metrics from a yaml file where the metrics are specified as dictionaries.
        """
        self.wandb = WandbConfig(**self.wandb)
