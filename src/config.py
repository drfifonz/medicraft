from pathlib import Path

from torch import cuda, device

DATASET_FILE_PATH = Path("datasets/ophthal_anonym/dataset.csv")

DEVICE = device("cuda" if cuda.is_available() else "cpu")

WANDB_PRJ_NAME_GENERATE_SAMPLES = "opthal_anonymized_datasets"
WANDB_PRJ_NAME_CLASSIFICATION = "medicraft-classification3"
WANDB_PRJ_NAME_TRAIN_GENERATOR = "medicraft"
