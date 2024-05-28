import logging
import re
from datetime import datetime
from enum import Enum
from pathlib import Path

import lightning as pl
import pandas as pd
import torch
import torch.nn as nn
from denoising_diffusion_pytorch import Unet
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms as T

import pipeline.blocks as pipeline_blocks
import wandb
from datasets import EyeScans, OpthalAnonymizedDataset, get_csv_dataset
from generate_samples import generate_samples as generate

# from config import DEVICE
# from datasets import OpthalAnonymizedDataset
from models import GaussianDiffusion, ResNetClassifier
from pipeline.parser import parse_config, read_config_file
from trackers import ImagePredictionLogger
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
            total_steps = -1
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
        print(f"{config=}")
        print("Generating samples")

        unet_config = models_config.unet
        diffusion_config = models_config.diffusion

        diffusion = self.__get_diffusion_model(image_size, diffusion_config, unet_config)
        print("Diffusion loaded")

        raise NotImplementedError("Generating samples")  # TODO remove
        if config.wandb:
            wandb.init(
                project="opthal_anonymized_datasets",
                tags=["opthal_anonymized", "generate_dataset"],
                mode="offline",
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
        Validate the model
        """
        print(f"{config=}")

        print(models_config)

        if config.classification:
            # print("Running classification experiment")
            # print(config.classification)
            # raise NotImplementedError("Running classification experiment")
            self.__run_classification_experiment(config.classification, models_config)
        raise NotImplementedError("Validating model")

    def run(self, verbose: bool = False):
        """
        Run the pipeline
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
        Run the classification experiment
        """
        classifier_config = models_config.classifier
        print(f"{classifier_config=}")
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
        # raise NotImplementedError("Run classification experiment")
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
        print(classifier_config)
        model = self.__get_classifier_model(config, classifier_config)
        wandb_logger = WandbLogger(
            project="medicraft-classification",
            mode="offline",
            job_type="train",
            tags=config.logger_tags,
        )

        early_stop_callback = EarlyStopping(monitor="val_loss")
        progressbar_callback = TQDMProgressBar()
        # Samples required by the custom ImagePredictionLogger callback to log image predictions.
        val_samples = next(iter(data_module.val_dataloader()))

        trainer = pl.Trainer(
            max_epochs=config.epochs,
            # progress_bar_refresh_rate=20,
            # gpus=1,
            logger=wandb_logger,
            callbacks=[early_stop_callback, ImagePredictionLogger(val_samples), progressbar_callback],
            # checkpoint_callback=checkpoint_callback,
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
        Load the configuration
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

    def __get_classifier_model(
        self, config: pipeline_blocks.ClassificationDTO, classifier_config: dict
    ) -> pl.LightningModule:
        def get_loss_fn(loss_fn_name: str) -> nn.Module:
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
