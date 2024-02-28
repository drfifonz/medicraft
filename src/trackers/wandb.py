import math
from collections.abc import Sequence

import torch
import wandb
from torch import nn
from torchvision import transforms
from torchvision.utils import make_grid


def is_square(num: int) -> bool:
    return math.isqrt(num) ** 2 == num


class WandbTracker:
    def __init__(self, project_name: str, hyperparameters: dict, tags: Sequence, group: str) -> None:
        # wandb.init(project=project_name, name=experiment_name)
        # self.run = wandb.run
        self.step = 0
        print("WandbTracker initialized")
        print(f"Project name: {project_name}")
        wandb.init(
            project=project_name,
            config=hyperparameters,
            group=group,
            tags=tags if tags else None,
        )
        assert wandb.run is not None
        wandb.define_metric("*", step_metric="global_step")

    def observe_model(self, model: nn.Module, log_freq: int = 1000) -> None:
        wandb.watch(model, log_freq=log_freq)

    def log(self, metrics: dict) -> None:
        metrics["global_step"] = self.step
        wandb.log(metrics)

    def log_images(self, images: torch.Tensor) -> None:
        assert is_square(len(images)), "Number of images must be a square number"
        grid = wandb.Image(make_grid(images, nrow=int(math.sqrt(len(images)))), caption=f"sample-grid-{self.step}")
        images = [transforms.ToPILImage()(image.cpu()) for image in images]
        images = [wandb.Image(image, caption=f"sample-{i}-{self.step}") for i, image in enumerate(images)]
        images.insert(0, grid)
        wandb.log(
            {
                "images": images,
                "global_step": self.step,
            }
        )

    def finish(self):
        self.run.finish()

    def get_experiment_name(self) -> str:
        return wandb.run.id

    def save_model(self, model_path: str) -> None:
        wandb.save(model_path)

    def update_step(self) -> None:
        self.step += 1
