"""
Configuration file for the Medicraft project.

This file contains various configuration parameters used throughout the project.

Attributes:
    DATASET_FILE_PATH (Path): The file path to the dataset CSV file.
    DEV_DEBUG (bool): Whether to enable debug mode.
    DEVICE (torch.device): The device to be used for computation (cuda if available, otherwise cpu).
    WANDB_PRJ_NAME_GENERATE_SAMPLES (str): The project name for generating samples.
    WANDB_PRJ_NAME_CLASSIFICATION (str): The project name for classification.
    WANDB_PRJ_NAME_TRAIN_GENERATOR (str): The project name for training the generator.
"""
from pathlib import Path

from torch import cuda, device

DATASET_FILE_PATH = Path("datasets/ophthal_anonym/dataset.csv")

DEV_DEBUG = True

DEVICE = device("cuda" if cuda.is_available() else "cpu")

WANDB_PRJ_NAME_GENERATE_SAMPLES = "opthal_anonymized_datasets"
WANDB_PRJ_NAME_CLASSIFICATION = "medicraft-classification3"
WANDB_PRJ_NAME_TRAIN_GENERATOR = "medicraft"
