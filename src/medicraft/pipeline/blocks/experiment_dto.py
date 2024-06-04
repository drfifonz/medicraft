from pathlib import Path
from typing import Literal, Optional, Union

from pipeline.blocks.models_dto import ModelsDTO
from pydantic import BaseModel, ConfigDict, computed_field, field_validator

TRAIN_GENERATOR = "train_generator"
GENERATE_SAMPLES = "generate_samples"
VALIDATE = "validate"
FOO = "foo"


class LoopObjectDTO(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: str
    repeat: bool = True

    @field_validator("name")
    def name_validator(cls, v):
        if v not in [TRAIN_GENERATOR, GENERATE_SAMPLES, VALIDATE, FOO]:
            raise ValueError("Undefined training loop object")
        return v.title()


class TrainGeneratorDTO(LoopObjectDTO):
    batch_size: int
    lr: float
    save_and_sample_every: int
    num_steps: int
    results_dir: str
    gradient_accumulate_every: int = 4

    experiment_id: Optional[str] = None
    copy_results_to: Optional[str] = None
    start_from_checkpoint: Optional[str] = None
    dataset_split_type: Literal["train", "val", "test"] = "train"
    diagnosis: Literal["precancerous", "fluid", "benign", "reference"]

    @field_validator("name")
    def name_validator(cls, v):
        if v != TRAIN_GENERATOR:
            raise ValueError("name must be 'train_generator'")
        return v.title()


class GenerateSamplesDTO(LoopObjectDTO):
    num_samples: int
    batch_size: int
    wandb: bool = True

    model_version: str
    base_on: str
    results_dir: str
    copy_results_to: Optional[str] = None
    checkpoint: Optional[str] = None
    relative_dataset_results_dir: str = "dataset"

    @computed_field
    @property
    def checkpoint_path(self) -> str:
        if self.checkpoint:
            return self.checkpoint
        return str(Path(self.results_dir) / self.base_on / f"{self.model_version}.pt")

    @computed_field
    @property
    def generete_samples_dir(self) -> str:
        return str(Path(self.results_dir) / self.relative_dataset_results_dir / self.base_on)

    @field_validator("name")
    def name_validator(cls, v):
        if v != GENERATE_SAMPLES:
            raise ValueError(f"name must be {GENERATE_SAMPLES}")
        return v.title()


class FooDTO(LoopObjectDTO):
    foo: str = "foo"

    @field_validator("name")
    def name_validator(cls, v):
        if v != FOO:
            raise ValueError(f"name must be {FOO}")
        return v.title()


class ClassificationDTO(BaseModel):
    epochs: int
    lr: float
    loss_multiply: float = 1.0
    ratio: list[float] = [0.8, 0.2]
    class_names: list[str] = ["fluid", "benign", "precancerous", "reference"]

    train_data_type: Literal["real", "synthetic"] = "synthetic"

    train_dataset_dir: str
    val_dataset_dir: Optional[str] = None
    test_dataset_dir: str

    loss_fn: str = "cross_entropy"
    num_workers: int = 4
    batch_size: int = 32
    min_epochs: int = 5
    log_every_n_steps: int = 10
    logger_tags: Optional[list[str]] = None
    logger_experiment_name: Optional[str] = None
    offline: bool = False

    results_dir: str

    @computed_field
    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class ValidateDTO(LoopObjectDTO):
    results_dir: str
    copy_results_to: Optional[str] = None

    classification: Optional[ClassificationDTO] = None

    @field_validator("name")
    def name_validator(cls, v):
        if v != VALIDATE:
            raise ValueError(f"name must be {VALIDATE}")
        return v.title()

    @field_validator("classification", mode="before")
    @classmethod
    def classification_validator(cls, v, values):
        v["results_dir"] = values.data.get("results_dir", ".results")
        return ClassificationDTO(**v)


class ExperimentDTO(BaseModel):
    total_steps: int
    image_size: list[int]
    models: ModelsDTO
    results_dir: str
    copy_results_to: Optional[str] = None
    loop: list[Union[TrainGeneratorDTO, GenerateSamplesDTO, ValidateDTO, FooDTO]]

    @field_validator("loop", mode="before")
    @classmethod
    def loop_validator(cls, v, values):
        res = []
        for loop_obj in v:
            loop_obj["results_dir"] = values.data.get("results_dir", ".results")
            loop_obj["copy_results_to"] = values.data.get("copy_results_to", None)
            if loop_obj["name"] == TRAIN_GENERATOR:
                res.append(TrainGeneratorDTO(**loop_obj))
            if loop_obj["name"] == GENERATE_SAMPLES:
                res.append(GenerateSamplesDTO(**loop_obj))
            if loop_obj["name"] == VALIDATE:
                res.append(ValidateDTO(**loop_obj))
            if loop_obj["name"] == FOO:
                res.append(FooDTO(**loop_obj))
        return res
