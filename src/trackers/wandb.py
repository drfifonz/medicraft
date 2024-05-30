import logging
import math

import torch
import wandb
from torch import nn
from torchvision import transforms
from torchvision.utils import make_grid


def is_square(num: int) -> bool:
    """
    Check if a number is a perfect square.

    Args:
        num (int): The number to be checked.

    Returns:
        bool: True if the number is a perfect square, False otherwise.
    """
    return math.isqrt(num) ** 2 == num


class WandbTracker:
    def __init__(self, project_name: str, hyperparameters: dict, tags: list, group: str, *args, **kwargs) -> None:
        """
        Initialize the WandbTracker.

        :param project_name: The name of the project.
        :type project_name: str
        :param hyperparameters: The hyperparameters for the project.
        :type hyperparameters: dict
        :param tags: The tags for the project.
        :type tags: list
        :param group: The group for the project.
        :type group: str
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        """

        self.step = 0
        logging.info("WandbTracker initialized")
        print(f"Project name: {project_name}")
        wandb.init(
            project=project_name,
            config=hyperparameters,
            group=group,
            tags=tags if tags else None,
            id=kwargs.get("id", None),
            resume=kwargs.get("resume", None),
            mode=kwargs.get("mode", "online"),
        )
        assert wandb.run is not None
        wandb.define_metric("*", step_metric="global_step")

    def observe_model(self, model: nn.Module, log_freq: int = 1000) -> None:
        """
        Observes the given model using Weights & Biases (wandb) library.

        Args:
            model (nn.Module): The model to be observed.
            log_freq (int, optional): The frequency at which to log the model. Defaults to 1000.
        """
        wandb.watch(model, log_freq=log_freq)

    def log(self, metrics: dict) -> None:
        """
        Logs the given metrics using WandB.

        Args:
            metrics (dict): A dictionary containing the metrics to be logged.

        Returns:
            None
        """
        metrics["global_step"] = self.step
        wandb.log(metrics)

    def log_images(self, images: torch.Tensor) -> None:
        """
        Logs a grid of images and individual images to WandB.

        Args:
            images (torch.Tensor): A tensor containing the images to be logged.

        Raises:
            AssertionError: If the number of images is not a square number.
        """

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
        """
        Finish the tracking run and save the results.
        """
        self.run.finish()

    def get_experiment_name(self) -> str:
        """
        Get the name of the current experiment.

        :return: The name of the experiment.
        :rtype: str
        """
        return wandb.run.id

    def save_model(self, model_path: str) -> None:
        """
        Save the model to the specified path using wandb.save.

        :param model_path: The path where the model should be saved.
        :type model_path: str
        """
        wandb.save(model_path)

    def update_step(self) -> None:
        """
        Increments the step counter by 1.
        """
        self.step += 1
