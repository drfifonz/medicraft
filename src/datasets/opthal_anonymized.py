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
    Load dataset from csv file and split it into train and validation sets from previosly splitted train/test dataset.
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
    def __init__(
        self,
        diagnosis: Literal["precancerous", "fluid", "benign", "reference"],
        df: pd.DataFrame,
        images_dir: str | Path,
        image_size: tuple[int, int] = (256, 512),
        transform: nn.Module = None,
        extension: list[str] = [".png", ".jpg", ".jpeg"],  # TODO use it
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
                    # T.RandomHorizontalFlip(),
                    # T.normalize(mean=[0.5,], std=[0.5,])
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


if __name__ == "__main__":
    DATASET_FILE_PATH = Path("data/datasets/ophthal_anonym-2024-04-22/new_df.csv")
    IMAGE_FILE_PATH = Path("data/datasets/ophthal_anonym-2024-04-22/images")
    df_dataset = get_csv_dataset(DATASET_FILE_PATH, val_size=0.1 / 0.9)

    train_dataset = OpthalAnonymizedDataset("reference", df_dataset["train"], IMAGE_FILE_PATH)
    test_dataset = OpthalAnonymizedDataset("reference", df_dataset["test"], IMAGE_FILE_PATH)
    val_dataset = OpthalAnonymizedDataset("reference", df_dataset["val"], IMAGE_FILE_PATH)
    # im_tensor = next(iter())
    print(len(train_dataset))
    print(len(test_dataset))
    print(len(val_dataset))

    # print(dataset.__get_df_by_diagnosis("reference"))
    # print(dataset.__get_df_by_diagnosis("precancerous"))
    # print(dataset.__get_df_by_diagnosis("fluid"))
    # print(dataset.__get_df_by_diagnosis("benign"))
