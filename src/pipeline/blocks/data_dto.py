from pathlib import Path

from pydantic import BaseModel, computed_field, field_validator


class DataDTO(BaseModel):
    csv_file_path: str
    validation_split: float = 0.2

    split_seed: int = 42

    @field_validator("validation_split", mode="before")
    @classmethod
    def validation_split_validator(cls, v):
        if isinstance(v, float) and 0 <= v < 1:
            return v
        elif isinstance(v, int) and 0 <= v < 100:
            return v / 100
        elif isinstance(v, str) and v.endswith("%"):
            return int(v[:-1]) / 100
        elif isinstance(v, str):
            return eval(v)
        else:
            raise ValueError("validation_split must be a float between 0 and 1")

    @computed_field
    @property
    def images_directory(self) -> str:
        return str(Path(self.csv_file_path).parent / "images")
