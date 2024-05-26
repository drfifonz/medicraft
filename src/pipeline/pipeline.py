import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

# import pandas as pd
import pandas as pd
import torch
import torch.nn as nn
from denoising_diffusion_pytorch import Unet
from torchvision import transforms as T

import pipeline.blocks as pipeline_blocks
from datasets import OpthalAnonymizedDataset, get_csv_dataset

# from config import DEVICE
# from datasets import OpthalAnonymizedDataset
from models import GaussianDiffusion
from pipeline.parser import parse_config, read_config_file
from trainers import Trainer
from utils import copy_results_directory
from utils.transforms import HorizontalCenterCrop

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
    images_directory: str
    __df: pd.DataFrame
    __image_size: list[int]

    runned_steps: int = 0
    train_dataset: torch.utils.data.Dataset
    val_dataset: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset

    def load_data(self, config: pipeline_blocks.DataDTO) -> None:
        logging.debug(f"{config=}")
        self.images_directory = config.images_directory

        self.__df = get_csv_dataset(
            filepath=config.csv_file_path,
            val_size=config.validation_split,
            seed=config.split_seed,
        )

    def train_generator(
        self, config: pipeline_blocks.TrainGeneratorDTO, models_config: dict, image_size: list[int]
    ) -> None:
        """
        Train the generator model
        """
        unet_config = models_config.unet
        diffusion_config = models_config.diffusion

        diffusion = self.__get_diffusion_model(image_size, diffusion_config, unet_config)

        logging.debug(f"{config.dataset_split_type=}")
        logging.debug(f"{config.batch_size=}")
        dataset = OpthalAnonymizedDataset(
            diagnosis=config.diagnosis,
            df=self.__df[config.dataset_split_type],
            images_dir=self.images_directory,
            transform=self.transform,
            convert_image_to="L",
        )
        print("---" * 12)
        print(config)
        print("---" * 12)

        if config.experiment_id:
            results_folder = Path(config.results_dir) / config.experiment_id / config.diagnosis
        else:
            results_folder = Path(config.results_dir) / config.diagnosis

        # raise NotImplementedError("dataset loaded")
        trainer = Trainer(  # noqa : F841
            diffusion_model=diffusion,
            folder=self.images_directory,
            dataset=dataset,
            train_batch_size=config.batch_size,
            train_lr=config.lr,
            save_and_sample_every=config.save_and_sample_every,
            # save_and_sample_every=10,
            results_folder=results_folder,  # TODO CHANGE THAT
            train_num_steps=config.num_steps,
            gradient_accumulate_every=config.gradient_accumulate_every,  # gradient accumulation steps
            ema_decay=0.995,  # exponential moving average decay
            amp=True,  # turn on mixed precision
            num_samples=9,  # number of samples to save
            calculate_fid=False,  # calculate FID during sampling
            tracker="wandb",
            tracker_kwargs={
                "tags": [config.diagnosis, "opthal_anonymized"],
                "mode": "offline",
            },
        )
        trainer.train()
        logging.info("Training completed successfully.")

        if config.copy_results_to:
            logging.info("Copying results...")
            copy_results_directory(
                results_folder,
                Path(config.copy_results_to) / config.experiment_id if config.experiment_id else config.copy_results_to,
            )
            logging.info("Results copied successfully.")

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

    def run(self, verbose: bool = False):
        """
        Run the pipeline
        """

        if verbose:
            self.__set_logging_level(
                level=10,
                save_to_file=False,
            )

        if self.config is None:
            raise ValueError("Configuration not loaded")

        # load data
        logging.info("Loading data...")
        self.load_data(self.config.get(PipelineBlocks.data.name))
        logging.info("Data loaded successfully.")

        logging.info("Training...")
        self.train(self.config.get(PipelineBlocks.training.name))

    def load_config(self, config_file: str | Path = "config.yml") -> None:
        """
        Load the configuration
        """
        config = read_config_file(config_file)
        self.config = parse_config(config)
        self.__image_size = self.config.get(PipelineBlocks.general.name).image_size
        logging.info("Configuration parsed successfully.")

    def __get_dataset(self, type: Literal["train", "val", "test"]):
        match type:
            case "train":
                return self.train_dataset
            case "val":
                return self.val_dataset
            case "test":
                return self.test_dataset

    @property
    def transform(self) -> T.Compose:
        return T.Compose(
            [
                HorizontalCenterCrop(512),
                T.Resize(self.__image_size),
                T.RandomHorizontalFlip(),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
            ]
        )

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

    def __set_logging_level(
        self,
        level: int = 10,
        save_to_file: bool = False,
        filename: str = "run.log",
    ) -> None:
        """
        Set the logging level
        10: DEBUG
        20: INFO
        30: WARNING
        40: ERROR
        50: CRITICAL
        """
        filename = Path(filename)

        current_date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        filename = filename.parent / f"{current_date}_{filename.name}"

        handlers = [logging.StreamHandler()]
        if save_to_file:
            handlers.append(logging.FileHandler(filename))

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
            handlers=handlers,
        )
