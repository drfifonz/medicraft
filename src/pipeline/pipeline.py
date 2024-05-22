import logging
from enum import Enum
from pathlib import Path

# import pandas as pd
import torch
import torch.nn as nn
from denoising_diffusion_pytorch import Unet

import pipeline.blocks as pipeline_blocks

# from config import DEVICE
# from datasets import OpthalAnonymizedDataset
# from datasets.opthal_anonymized import get_csv_dataset
from models import GaussianDiffusion
from pipeline.parser import parse_config, read_config_file
from trainers import Trainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PipelineBlocks(Enum):
    """
    Enum for the pipeline blocks
    """

    general = "general"
    data = "data"
    training = "training"
    output = "output"


class Pipeline:
    """
    A class to represent a pipeline
    """

    config: dict

    def __init__(self):
        self.runned_steps = 0

    def load_data(self, config):
        print(f"{config=}")
        pass

    def train_generator(
        self, config: pipeline_blocks.TrainGeneratorDTO, models_config: dict, image_size: list[int]
    ) -> None:
        """
        Train the generator model
        """
        unet_config = models_config.unet
        diffusion_config = models_config.diffusion

        diffusion = self.__get_diffusion_model(image_size, diffusion_config, unet_config)

        raise ValueError("Not implemented")
        trainer = Trainer(  # noqa : F841
            diffusion,
            str(self.csv_data_file.parent / "images"),
            dataset=self.dataset,
            train_batch_size=config.batch_size,
            train_lr=config.lr,
            save_and_sample_every=2000,
            # save_and_sample_every=10,
            results_folder="./.results/reference",
            train_num_steps=100_000,  # total training steps
            break_every_steps=3000,  # TODO add that
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

    def train(self, config) -> None:
        """
        Train the pipeline with specified configuration
        """
        train_loop_blocks: list = config.train_loop

        only_once_blocks = [block for block in train_loop_blocks if block.repeat == False]

        total_steps = config.total_steps

        models_config = config.models
        image_size = config.image_size

        while total_steps > self.runned_steps:
            print(f"Step {self.runned_steps}")
            for block in train_loop_blocks:
                if block not in only_once_blocks and not block.repeat:
                    continue
                if block.name.lower() == pipeline_blocks.TRAIN_GENERATOR:
                    self.train_generator(block, models_config, image_size)
                elif block.name.lower() == pipeline_blocks.GENERATE_SAMPLES:
                    self.generate_samples(block)
                elif block.name.lower() == pipeline_blocks.VALIDATE:
                    self.validate(block)
                elif block.name.lower() == pipeline_blocks.FOO:
                    print("foo")
                    self.foo(block)
                if not block.repeat:
                    only_once_blocks.remove(block)

    def foo(self, config: pipeline_blocks.FooDTO):
        """
        Foo
        """
        print(f"{config.foo=}")
        self.runned_steps += 10

    def generate_samples(self, config: pipeline_blocks.GenerateSamplesDTO):
        """
        Generate dataset
        """
        print("Generating samples")

    def validate(self, config: pipeline_blocks.ValidateDTO):
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

    def run(self):
        """
        Run the pipeline
        """
        if self.config is None:
            raise ValueError("Configuration not loaded")

        # load data

        self.load_data(self.config.get(PipelineBlocks.data.name))

        self.train(self.config.get(PipelineBlocks.training.name))

    def load_config(self, config_file: str | Path = "config.yml") -> None:
        """
        Load the configuration
        """
        config = read_config_file(config_file)
        self.config = parse_config(config)
        logging.info("Configuration parsed successfully.")

    def __get_unet_model(self, dim: int, dim_mults: list[int], channels: int) -> nn.Module:
        """
        Create a Unet model
        """
        model = Unet(
            dim=dim,
            dim_mults=dim_mults,
            channels=channels,
        )
        model.to(device=DEVICE)
        return model

    def __get_diffusion_model(self, image_size: list[int], diffusion_config: dict, unet_config: dict) -> nn.Module:
        """
        Create a Gaussian diffusion model
        """
        model = self.__get_unet_model(**unet_config)
        diffusion = GaussianDiffusion(
            model,
            image_size=image_size,
            **diffusion_config
            # loss_type = 'l1'    # L1 or L2
        )
        diffusion.to(device=DEVICE)
        logging.info(f"Model loaded to {DEVICE} device.")
        return diffusion
