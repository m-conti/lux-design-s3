
from typing import Any


class WandbLogger:
    def __init__(self,config: dict[str, Any], name: str):
        import wandb
        self.run = wandb.init(project=config['project'], name=name)
        self.run.config.update(config)
        self.run.config.update({'run_name': name})

    def log(self, metrics: dict[str, float], step: int):
        self.run.log(metrics, step=step)

    def finish(self):
        self.run.finish()

    def __del__(self):
        self.finish()