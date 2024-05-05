from pathlib import Path

import torch

DATASET_FILE_PATH = Path("datasets/ophthal_anonym-2024-04-22/dataset.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
