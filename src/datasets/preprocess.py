from pathlib import Path
from typing import Literal

import pandas as pd
from PIL import Image


def crop_image(
    image: Image, new_size: tuple[int, int], orientation: Literal["center", "left", "right"] = "center"
) -> Image:
    """
    Crop image to new_size with orientation
    """
    width, height = image.size

    left = (width - new_size[0]) // 2
    right = (width + new_size[0]) // 2
    top = (height - new_size[1]) // 2
    bottom = (height + new_size[1]) // 2

    match orientation:
        case "center":
            return image.crop((left, top, right, bottom))
        case "left":
            return image.crop((0, top, new_size[0], bottom))
        case "right":
            return image.crop((width - new_size[0], top, width, bottom))
        case _:
            raise ValueError("Invalid orientation")


def load_dataset_csv_file(file_path: str | Path) -> pd.DataFrame:
    if isinstance(file_path, str):
        file_path = Path(file_path)

    images_dir = file_path.parent / "images"

    df = pd.read_csv(file_path)
    df["file_path"] = images_dir / df["filename"]

    return df
