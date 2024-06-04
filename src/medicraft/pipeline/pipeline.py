import logging
import re
from datetime import datetime
from enum import Enum
from pathlib import Path

import config as cfg
import lightning as pl
import pandas as pd
import pipeline.blocks as pipeline_blocks
import torch
import torch.nn as nn
import wandb
from datasets import EyeScans, OpthalAnonymizedDataset, get_csv_dataset
from denoising_diffusion_pytorch import Unet
from generate_samples import generate_samples as generate
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from models import GaussianDiffusion, ResNetClassifier
from pipeline.parser import parse_config, read_config_file
from torchvision import transforms as T
from trackers import ImagePredictionLogger
from trainers import Trainer
from utils import copy_results_directory
from utils.transforms import HorizontalCenterCrop


class PipelineBlocks(Enum):
    """
    Enum for the pipeline blocks

    :cvar general: Represents the general block
    :vartype general: str
    :cvar data: Represents the data block
    :vartype data: str
    :cvar experiment: Represents the experiment block
    :vartype experiment: str
    :cvar output: Represents the output block
    :vartype output: str
    """

    general = "general"
    data = "data"
    experiment = "experiment"
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

        :param config: Configuration for training the generator model
        :type config: pipeline_blocks.TrainGeneratorDTO
        :param models_config: Configuration for the generator and diffusion models
        :type models_config: dict
        :param image_size: Size of the input images
        :type image_size: list[int]
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

        if config.experiment_id:
            results_folder = Path(config.results_dir) / config.experiment_id / config.diagnosis
        else:
            results_folder = Path(config.results_dir) / config.diagnosis

        trainer = Trainer(  # noqa : F841
            diffusion_model=diffusion,
            folder=self.images_directory,
            dataset=dataset,
            train_batch_size=config.batch_size,
            train_lr=config.lr,
            save_and_sample_every=config.save_and_sample_every,
            # save_and_sample_every=10,
            results_folder=results_folder,
            train_num_steps=config.num_steps,
            gradient_accumulate_every=config.gradient_accumulate_every,  # gradient accumulation steps
            ema_decay=0.995,  # exponential moving average decay
            amp=True,  # turn on mixed precision
            num_samples=9,  # number of samples to save
            calculate_fid=False,  # calculate FID during sampling
            tracker="wandb",
            tracker_kwargs={
                "tags": [config.diagnosis, "opthal_anonymized"],
                "project_name": cfg.WANDB_PRJ_NAME_TRAIN_GENERATOR,
            },
        )
        if config.start_from_checkpoint:
            trainer.load(config.start_from_checkpoint)
            logging.info(f"Model loaded from {config.start_from_checkpoint}.")
        trainer.train()
        logging.info("Training completed successfully.")
        self.runned_steps += config.num_steps

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
        loop_blocks: list = config.loop

        only_once_blocks = [block for block in loop_blocks if not block.repeat]

        total_steps = config.total_steps

        models_config = config.models
        image_size = config.image_size

        if total_steps == 0 and self.runned_steps == 0:
            logging.info("Running 1 iteration of the pipeline")
            self.runned_steps = -1
        while total_steps > self.runned_steps:
            print(f"Step {self.runned_steps}")
            for block in loop_blocks:
                if block not in only_once_blocks and not block.repeat:
                    continue
                if block.name.lower() == pipeline_blocks.TRAIN_GENERATOR:
                    self.train_generator(block, models_config, image_size)
                elif block.name.lower() == pipeline_blocks.GENERATE_SAMPLES:
                    self.generate_samples(block, models_config, image_size)
                elif block.name.lower() == pipeline_blocks.VALIDATE:
                    self.validate(block, models_config)
                elif block.name.lower() == pipeline_blocks.FOO:
                    print("foo")
                    self.foo(block)
                if not block.repeat:
                    only_once_blocks.remove(block)
            if total_steps == 0:
                break

    def foo(self, config: pipeline_blocks.FooDTO):
        """
        Foo
        """
        print(f"{config.foo=}")
        self.runned_steps += 10

    def generate_samples(self, config: pipeline_blocks.GenerateSamplesDTO, models_config: dict, image_size: list[int]):
        """
        Generate samples
        """
        logging.info("Generating samples")

        unet_config = models_config.unet
        diffusion_config = models_config.diffusion

        diffusion = self.__get_diffusion_model(image_size, diffusion_config, unet_config)
        print("Diffusion loaded")

        if config.wandb:
            wandb.init(
                project=cfg.WANDB_PRJ_NAME_GENERATE_SAMPLES,
                tags=["opthal_anonymized", "generate_dataset"],
            )

        diffusion.load_state_dict(torch.load(config.checkpoint_path)["model"])
        # checkpoint = torch.load(config.checkpoint_path)
        logging.info(f"Model loaded from {config.checkpoint_path}.")

        generate(
            diffusion_model=diffusion,
            results_dir=config.generete_samples_dir,
            num_samples=config.num_samples,
            batch_size=config.batch_size,
        )

        if config.copy_results_to:
            logging.info("Copying results...")
            copy_results_directory(
                config.generete_samples_dir,
                str(Path(config.copy_results_to) / config.relative_dataset_results_dir / config.base_on),
            )
            logging.info("Results copied successfully.")

        raise NotImplementedError("Generating samples")

    def validate(self, config: pipeline_blocks.ValidateDTO, models_config: dict):
        """
        Validate experiment block

        :param config: The configuration for validation
        :type config: pipeline_blocks.ValidateDTO
        :param models_config: The configuration for the models
        :type models_config: dict
        """
        if config.classification:
            self.__run_classification_experiment(config.classification, models_config)

    def run(self, verbose: bool = False):
        """
        Run the pipeline.

        :param verbose: Whether to enable verbose logging (default: False)
        :type verbose: bool
        :raises ValueError: If configuration is not loaded
        """
        if verbose:
            self.__set_logging_level(
                level=20,
                save_to_file=False,
            )

        if self.config is None:
            raise ValueError("Configuration not loaded")

        # load data
        logging.info("Loading data...")
        self.load_data(self.config.get(PipelineBlocks.data.name))
        logging.info("Data loaded successfully.")

        logging.info("Training...")
        self.train(self.config.get(PipelineBlocks.experiment.name))

    def __run_classification_experiment(self, config: pipeline_blocks.ClassificationDTO, models_config: dict) -> None:
        """
        Run the classification experiment.

        Args:
            config (pipeline_blocks.ClassificationDTO): The configuration for the classification experiment.
            models_config (dict): The configuration for the models.

        Returns:
            None
        """
        classifier_config = models_config.classifier
        match config.train_data_type:
            case "real":
                is_real_train_data = True
            case "synthetic":
                is_real_train_data = False
            case _:
                raise ValueError(f"Unknown data type: {config.train_data_type}")

        train_dataset_dir = config.train_dataset_dir
        val_dataset_dir = config.val_dataset_dir
        test_dataset_dir = config.test_dataset_dir

        data_module = EyeScans(
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            ratio=config.ratio,
            real_word_data=is_real_train_data,
            train_data_dir=train_dataset_dir,
            val_data_dir=val_dataset_dir,
            test_dataset_dir=test_dataset_dir,
        )

        data_module.setup()
        logging.info("Data module setup completed successfully.")
        model = self.__get_classifier_model(config, classifier_config)
        wandb_logger = WandbLogger(
            project=cfg.WANDB_PRJ_NAME_CLASSIFICATION,
            id=config.logger_experiment_name,
            offline=config.offline,
            save_dir=Path(config.results_dir) / "classification-wandb",
            job_type="train",
            tags=config.logger_tags,
        )

        early_stop_callback = EarlyStopping(monitor="val_loss")
        progressbar_callback = TQDMProgressBar()

        val_samples = next(iter(data_module.val_dataloader()))

        trainer = pl.Trainer(
            min_epochs=config.min_epochs,
            max_epochs=config.epochs,
            logger=wandb_logger,
            callbacks=[early_stop_callback, ImagePredictionLogger(val_samples), progressbar_callback],
            enable_checkpointing=True,
            enable_progress_bar=True,
            log_every_n_steps=config.log_every_n_steps,
        )
        trainer.fit(model, data_module)
        trainer.test(model, data_module)

        wandb.finish()
        logging.info("Classification process completed successfully.")

    def load_config(self, config_file: str | Path = "config.yml") -> None:
        """
        Load the configuration.

        :param config_file: The path to the configuration file. Default is "config.yml".
        :type config_file: str or Path
        :return: None
        """
        config = read_config_file(config_file)
        self.config = parse_config(config)
        self.__image_size = self.config.get(PipelineBlocks.general.name).image_size
        logging.info("Configuration parsed successfully.")

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

        :param dim: The dimension of the model
        :type dim: int
        :param dim_mults: The dimension multipliers for each layer in the model
        :type dim_mults: list[int]
        :param channels: The number of input channels
        :type channels: int
        :return: The Unet model
        :rtype: nn.Module
        """
        model = Unet(
            dim=dim,
            dim_mults=dim_mults,
            channels=channels,
        )
        model.to(device=cfg.DEVICE)
        return model

    def __get_diffusion_model(self, image_size: list[int], diffusion_config: dict, unet_config: dict) -> nn.Module:
        """
        Create a Gaussian diffusion model

        :param image_size: The size of the input image
        :type image_size: list[int]
        :param diffusion_config: Configuration parameters for Gaussian diffusion
        :type diffusion_config: dict
        :param unet_config: Configuration parameters for the U-Net model
        :type unet_config: dict
        :return: The diffusion model
        :rtype: nn.Module
        """
        model = self.__get_unet_model(**unet_config)
        diffusion = GaussianDiffusion(
            model,
            image_size=image_size,
            **diffusion_config
            # loss_type = 'l1'    # L1 or L2
        )
        diffusion.to(device=cfg.DEVICE)
        logging.info(f"Model loaded to {cfg.DEVICE} device.")
        return diffusion

    def __get_classifier_model(
        self, config: pipeline_blocks.ClassificationDTO, classifier_config: dict
    ) -> pl.LightningModule:
        """
        Get the classifier model based on the provided configuration.

        Args:
            config (pipeline_blocks.ClassificationDTO): The configuration for the classifier.
            classifier_config (dict): The configuration for the specific classifier.

        Returns:
            pl.LightningModule: The classifier model.

        Raises:
            ValueError: If the architecture or loss function is unknown.
        """

        def get_loss_fn(loss_fn_name: str) -> nn.Module:
            """
            Get the loss function based on the provided name.

            Args:
                loss_fn_name (str): The name of the loss function.

            Returns:
                nn.Module: The loss function module.

            Raises:
                ValueError: If the loss function name is unknown.
            """
            if loss_fn_name.lower() == "cross_entropy":
                return nn.CrossEntropyLoss()
            elif loss_fn_name.lower() == "nll":
                return nn.NLLLoss()
            else:
                raise ValueError(f"Unknown loss function: {loss_fn_name}")

        architecture: str = classifier_config.get("architecture")
        if re.match(r"resnet", architecture.lower()):
            model = ResNetClassifier(
                architecture=architecture,
                num_classes=config.num_classes,
                loss_fn=get_loss_fn(config.loss_fn),
                pretrained=False,
                learning_rate=config.lr,
                loss_multiply=config.loss_multiply,
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        return model

    def __set_logging_level(
        self,
        level: int = 10,
        save_to_file: bool = False,
        filename: str = "run.log",
    ) -> None:
        """
        Set the logging level

        :param level: The logging level to set. Valid values are:
                      10: DEBUG
                      20: INFO
                      30: WARNING
                      40: ERROR
                      50: CRITICAL
        :type level: int
        :param save_to_file: Whether to save logs to a file. Defaults to False.
        :type save_to_file: bool
        :param filename: The name of the log file. Defaults to "run.log".
        :type filename: str
        :return: None
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
