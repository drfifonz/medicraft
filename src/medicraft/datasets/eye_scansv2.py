import logging
from pathlib import Path

import lightning as pl
import torch
import torchvision.transforms as T

from .generated_dataset import GeneratedOCTDataset


class EyeScansV2(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset_csv_file: str | Path,
        test_dataset_csv_file: str | Path | None = None,
        num_workers: int = 4,
        transforms: T.Compose = None,
    ) -> None:
        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.dataset_csv_file = dataset_csv_file
        self.test_dataset_csv_file = test_dataset_csv_file

        self.batch_size = batch_size

        self.transforms = (
            T.Compose(
                [
                    T.CenterCrop((256, 512)),
                    T.Resize((256, 512)),
                    T.ToTensor(),
                    T.Normalize((0.5,), (0.5,)),
                ]
            )
            if transforms is None
            else transforms
        )
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        """
        Set up the eye scans dataset for training, validation, and testing.

        Args:
            stage (str, optional): The stage of the dataset setup. Can be "fit" for training, "test" for testing,
                or None for both. Defaults to None.
        """

        datasets = self.__prepare_datasets()

        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["val"]
        self.test_dataset = datasets["test"]

        if stage == "fit" or stage is None:
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(len(self.train_dataset)))
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(len(self.val_dataset)))
        if stage == "test" or stage is None:
            self.test_dataset = torch.utils.data.Subset(self.test_dataset, range(len(self.test_dataset)))

    def __prepare_datasets(self) -> dict[torch.utils.data.Dataset]:
        """
        Prepare the synthetic dataset with real world data test set.
        """
        logging.info("Loading OCT dataset...")
        train_dataset = GeneratedOCTDataset(
            csv_file=self.dataset_csv_file,
            transforms=self.transforms,
            split_type="train",
        )
        val_dataset = GeneratedOCTDataset(
            csv_file=self.dataset_csv_file,
            transforms=self.transforms,
            split_type="val",
        )
        test_dataset = GeneratedOCTDataset(
            csv_file=(self.test_dataset_csv_file if self.test_dataset_csv_file else self.dataset_csv_file),
            transforms=self.transforms,
            split_type="test",
        )

        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
