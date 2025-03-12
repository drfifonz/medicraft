from pathlib import Path

import lightning as pl
import torch
import torchvision.transforms as T
from datasets.generated_dataset import GeneratedOCTDataset
from torchvision import datasets


class EyeScansV2(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        # real_word_data: bool,
        dataset_csv_file: str | Path,
        ratio: list[float] = [0.8, 0.1],
        seed: int = 42,
        train_data_dir: str | None = None,
        val_data_dir: str | None = None,
        test_dataset_dir: str | None = None,
        num_workers: int = 4,
        transforms: T.Compose = None,
    ) -> None:

        super().__init__()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        # self.real_word_data = real_word_data

        # self.train_data_dir = train_data_dir
        # self.val_data_dir = val_data_dir
        # self.test_dataset_dir = test_dataset_dir

        self.dataset_csv_file = dataset_csv_file

        self.batch_size = batch_size
        self.ratio = ratio
        self.seed = seed

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

        torch.manual_seed(self.seed)

        # if self.real_word_data:
        #     datasets = self._prepare_real_world_datasets()
        # else:
        #     if self.val_data_dir is not None:
        #         raise ValueError("Validation data directory can't be defined for synthetic data")
        #     datasets = self.__prepare_synthetic_datasets()

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
        dataset = datasets.ImageFolder(root=self.train_data_dir, transform=self.transforms)
        dataset = GeneratedOCTDataset(csv_file=self.dataset_csv_file, transform=self.transform)
        if len(self.ratio) == 2:
            train_size = int(self.ratio[0] * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            test_dataset = datasets.ImageFolder(root=self.test_dataset_dir, transform=self.transforms)
        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": self.__prepare_real_test_dataset(),
        }

    def __prepare_real_test_dataset(self) -> torch.utils.data.Dataset:
        return

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
