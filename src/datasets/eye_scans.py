import lightning as pl
import torch
import torchvision.transforms as transforms
from torchvision import datasets


class EyeScans(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        ratio: list[float] = [0.8, 0.1, 0.1],
        seed: int = 42,
        test_dataset_dir: str | None = None,
        num_workers: int = 4,
    ) -> None:
        """
        Initializes an instance of the EyeScans dataset.

        :param data_dir: The directory path where the dataset is located.
        :param batch_size: The batch size for data loading.
        :param ratio: The ratio of train, validation, and test data split. Default is [0.8, 0.1, 0.1].
        :param seed: The random seed for reproducibility. Default is 42.
        :param test_dataset_dir: The directory path for the test dataset. Default is None.
        :param num_workers: The number of worker threads for data loading. Default is 4.
        """

        super().__init__()
        self.data_dir = data_dir
        self.test_dataset_dir = test_dataset_dir
        self.batch_size = batch_size
        self.ratio = ratio
        self.seed = seed

        self.transforms = transforms.Compose(
            [
                transforms.Resize((256, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        torch.manual_seed(self.seed)

        dataset = datasets.ImageFolder(
            root=self.data_dir,
            transform=self.transforms,
        )

        train_size = int(self.ratio[0] * len(dataset))
        val_size = int(self.ratio[1] * len(dataset))
        if self.test_dataset_dir:
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            self.test_dataset = datasets.ImageFolder(root=self.test_dataset_dir, transform=self.transforms)
        else:
            test_size = len(dataset) - train_size - val_size

            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )

        if stage == "fit" or stage is None:
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(len(self.train_dataset)))
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(len(self.val_dataset)))
        if stage == "test" or stage is None:
            self.test_dataset = torch.utils.data.Subset(self.test_dataset, range(len(self.test_dataset)))

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
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
