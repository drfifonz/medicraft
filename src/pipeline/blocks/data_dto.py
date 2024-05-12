from pathlib import Path
from typing import Optional

from pydantic import BaseModel, computed_field


class DataDTO(BaseModel):
    csv_file_path: str
    seperate_validation_data: bool = True
    validation_split: float = 0.2 if seperate_validation_data else 0.0
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    # drop_last: bool = False

    split_seed: int = 42

    @computed_field
    @property
    def images_directory(self) -> str:
        return str(Path(self.csv_file_path).parent / "images")
