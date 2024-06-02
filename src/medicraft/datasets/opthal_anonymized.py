from functools import partial
from pathlib import Path
from typing import Literal

import pandas as pd
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import convert_image_to_fn, exists
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T


def get_csv_dataset(
    filepath: str | Path,
    val_size: float | None = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Load dataset from csv file and split it into train validation,test sets from previously splitted train/test dataset.

    :param filepath: The path to the CSV file containing the dataset.
    :type filepath: str or Path
    :param val_size: The proportion of the dataset to be used for validation. Default is 0.1 (10%).
    :type val_size: float or None
    :param seed: The random seed for reproducible train/validation split. Default is 42.
    :type seed: int
    :return: A dictionary containing the train, validation, and test sets as pandas DataFrames.
             If val_size is None, only the train and test sets are returned.
    :rtype: dict[str, pd.DataFrame]
    """
    filepath = Path(filepath) if isinstance(filepath, str) else filepath

    df = pd.read_csv(filepath)
    df = df[df["image_type"] == "OCT"]

    result = {}
    if val_size is not None:
        train_df, val_df = train_test_split(df, test_size=val_size, random_state=seed)
        result["val"] = val_df
    else:
        train_df = df[df["split"] == "train"]

    result["train"] = train_df
    result["test"] = df[df["split"] == "test"]

    return result


class OpthalAnonymizedDataset(Dataset):
    """
    Dataset class for OpthalAnonymizedDataset.

    Args:
        diagnosis (Literal["precancerous", "fluid", "benign", "reference"]): The diagnosis category for the dataset.
        df (pd.DataFrame): The DataFrame containing the dataset information.
        images_dir (str | Path): The directory path where the images are stored.
        image_size (tuple[int, int], optional): The desired size of the images. Defaults to (256, 512).
        transform (nn.Module, optional): The transformation to apply to the images. Defaults to None.
        convert_image_to (Any, optional): The function or transformation to convert the images to a specific format. Defaults to None.
        seed (int, optional): The random seed for shuffling the dataset. Defaults to None.
    """

    def __init__(
        self,
        diagnosis: Literal["precancerous", "fluid", "benign", "reference"],
        df: pd.DataFrame,
        images_dir: str | Path,
        image_size: tuple[int, int] = (256, 512),
        transform: nn.Module = None,
        convert_image_to=None,
        seed: int = None,
    ):
        self.df = df
        self.images_dir = Path(images_dir) if isinstance(images_dir, str) else images_dir
        self.df["image_path"] = self.df["filename"].apply(lambda x: self.images_dir / x)

        self.diagnosis = diagnosis
        self.diagnosis_df = self.__get_df_by_diagnosis(self.diagnosis)
        self.classes = [*self.df["diagnosis"].unique(), "reference"]

        maybe_convert_fn = (
            partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        )  # TODO simplify or rename

        self.transform = (
            T.Compose(
                [
                    T.Lambda(maybe_convert_fn),
                    T.CenterCrop(image_size),
                    T.Resize(image_size),
                    T.Grayscale(num_output_channels=1),
                    T.ToTensor(),
                ]
            )
            if transform is None
            else transform
        )

    def __get_df_by_diagnosis(self, diagnosis: Literal["precancerous", "fluid", "benign", "reference"]):
        match diagnosis:
            case "precancerous":
                return self.df[self.df["reference_eye"] == False][self.df["diagnosis"] == "precancerous"]  # noqa: E712
            case "fluid":
                return self.df[self.df["reference_eye"] == False][self.df["diagnosis"] == "fluid"]  # noqa: E712
            case "benign":
                return self.df[self.df["reference_eye"] == False][self.df["diagnosis"] == "benign"]  # noqa: E712
            case "reference":
                return self.df[self.df["reference_eye"] == True]  # noqa: E712

    def __getitem__(self, idx: int):
        image_path = self.diagnosis_df.iloc[idx]["image_path"]
        image = Image.open(image_path)
        return self.transform(image)

    def __len__(self):
        return self.diagnosis_df.shape[0]
