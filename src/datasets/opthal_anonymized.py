from functools import partial
from pathlib import Path
from typing import Literal

import pandas as pd
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import convert_image_to_fn, exists
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T


class OpthalAnonymizedDataset(Dataset):
    def __init__(
        self,
        diagnosis: Literal["precancerous", "fluid", "benign", "reference"],
        csv_dataset_file: str | Path,
        image_size: tuple[int, int] = (256, 512),
        transform: nn.Module = None,  # TODO use it
        extension: list[str] = [".png", ".jpg", ".jpeg"],  # TODO use it
        convert_image_to=None,
    ):
        self.csv_dataset_file = Path(csv_dataset_file) if isinstance(csv_dataset_file, str) else csv_dataset_file
        self.images_dir = self.csv_dataset_file.parent / "images"

        df = pd.read_csv(csv_dataset_file)
        self.df = df[df["image_type"] == "OCT"]
        self.df["image_path"] = self.df["filename"].apply(lambda x: self.images_dir / x)

        self.diagnosis = diagnosis
        self.diagnosis_df = self.__get_df_by_diagnosis(self.diagnosis)
        self.classes = [*self.df["diagnosis"].unique(), "reference"]

        maybe_convert_fn = (
            partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        )  # TODO simplify

        self.transform = (
            T.Compose(
                [
                    T.Grayscale(num_output_channels=1),
                    T.Lambda(maybe_convert_fn),
                    T.CenterCrop(image_size),
                    T.Resize(image_size),
                    # T.RandomHorizontalFlip(),
                    # T.normalize(mean=[0.5], std=[0.5])
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
        # raise
        return self.transform(image)

    def __len__(self):
        return self.diagnosis_df.shape[0]


if __name__ == "__main__":
    DATASET_FILE_PATH = Path("datasets/ophthal_anonym-2024-04-22/dataset.csv")

    dataset = OpthalAnonymizedDataset(root=Path("data"), csv_dataset_file=DATASET_FILE_PATH)
    print(dataset.diagnosis)
    # print(dataset.__get_df_by_diagnosis("reference"))
    # print(dataset.__get_df_by_diagnosis("precancerous"))
    # print(dataset.__get_df_by_diagnosis("fluid"))
    # print(dataset.__get_df_by_diagnosis("benign"))
