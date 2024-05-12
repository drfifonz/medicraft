import logging
from enum import Enum
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from denoising_diffusion_pytorch import Unet

# from config import DEVICE
# from datasets import OpthalAnonymizedDataset
# from datasets.opthal_anonymized import get_csv_dataset
from models import GaussianDiffusion
from trainers import Trainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PipelineBlocks(Enum):
    """
    Enum for the pipeline blocks
    """

    LOAD_DATA = "load_data"
    TRAIN_MODEL = "train_model"
    VALIDATE_MODEL = "validate_model"
    GENERATE_DATASET = "generate_dataset"
    CALCULATE_DATASET_METRICS = "calculate_dataset_metrics"
    GET_DATASET_EMBEDDINGS = "get_dataset_embeddings"
    LOAD_DATASET_EMBEDDINGS = "load_dataset_embeddings"


class Pipeline:
    """
    A class to represent a pipeline
    """

    def __init__(self, csv_data_file: str):
        """
        Initialize the pipeline
        """
        self.dataset: dict[str, pd.DataFrame]
        self.csv_data_file = csv_data_file if isinstance(csv_data_file, Path) else Path(csv_data_file)

        self.pipeline = []

    def load_data(self, val_size: float | None = 0.1 / 0.9, seed: int = 42):
        """
        Load data from the data source
        """
        # self.dataset = get_csv_dataset(
        #     filepath=self.csv_data_file,
        #     val_size=val_size,
        #     seed=seed,
        # )

    def traing_generator(self, break_every_steps: int | None = None) -> None:
        """
        Train the model
        """
        diffusion = self.__get_diffusion_model()
        trainer = Trainer(  # noqa : F841
            diffusion,
            str(self.csv_data_file.parent / "images"),
            dataset=self.dataset,
            train_batch_size=4,
            train_lr=2e-4,
            save_and_sample_every=2000,
            # save_and_sample_every=10,
            results_folder="./.results/reference",
            train_num_steps=100_000,  # total training steps
            break_every_steps=break_every_steps,  # TODO add that
            gradient_accumulate_every=4,  # gradient accumulation steps
            ema_decay=0.995,  # exponential moving average decay
            amp=True,  # turn on mixed precision
            num_samples=9,  # number of samples to save
            calculate_fid=False,  # calculate FID during sampling
            tracker="wandb",
            tracker_kwargs={
                "tags": ["reference_eyes"],
                "mode": "online",
            },
        )
        trainer.train()

    def generate_dataset(self):
        """
        Generate dataset
        """
        pass

    def validate_model(self):
        """
        Validate the model
        """
        pass

    def calculate_dataset_metrics(self):
        """
        Calculate the metrics
        """
        pass

    def get_dataset_embeddings(self):
        """
        Get the embeddings
        """
        pass

    def load_dataset_embeddings(self):
        """
        Load the embeddings
        """
        pass

    def train(self, val_every_steps: int, total_steps: 100_000):
        """
        Train the pipeline
        """
        steps = 0
        while steps <= total_steps:
            self.traing_generator(break_every_steps=val_every_steps)
            self.generate_dataset()
            self.validate_model()

            steps += val_every_steps

        self.traing_generator()

    def run(self):
        """
        Run the pipeline
        """

        # training

    def __get_unet_model(self) -> nn.Module:
        """
        Create a UNet model
        """
        model = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=1,
        )
        model.to(device=DEVICE)
        return model

    def __get_diffusion_model(self) -> nn.Module:
        """
        Create a Gaussian diffusion model
        """
        model = self.__get_unet_model()
        diffusion = GaussianDiffusion(
            model,
            image_size=(512, 256)[::-1],
            timesteps=1000,  # number of steps #
            # timesteps=2,  # number of steps
            # loss_type = 'l1'    # L1 or L2
        )
        diffusion.to(device=DEVICE)
        logging.info(f"Model loaded to {DEVICE} device.")
        return diffusion


if __name__ == "__main__":
    pipeline = Pipeline()

    pipeline.run()
