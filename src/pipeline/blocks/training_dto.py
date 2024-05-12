from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, computed_field, field_validator

TRAIN_GENERATOR = "train_generator"
GENERATE_SAMPLES = "generate_samples"
VALIDATE = "validate"
FOO = "foo"


class LoopObjectDTO(BaseModel):
    name: str
    repeat: bool = False

    @field_validator("name")
    def name_validator(cls, v):
        if v not in [TRAIN_GENERATOR, GENERATE_SAMPLES, VALIDATE, FOO]:
            raise ValueError("Undefined training loop object")
        return v.title()


class TrainGeneratorDTO(LoopObjectDTO):
    batch_size: int = 8
    lr: float = 1e-3
    save_and_sample_every: int = 2000
    num_steps: int = 10_000
    results_dir: str

    @field_validator("name")
    def name_validator(cls, v):
        if v != TRAIN_GENERATOR:
            raise ValueError("name must be 'foo'")
        return v.title()


class GenerateSamplesDTO(LoopObjectDTO):
    num_samples: int = 100
    batch_size: int = 8

    save_dir: str

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


class ValidateDTO(LoopObjectDTO):
    fid: bool = False
    fid_args: Optional[dict] = None

    embeddings: bool = False
    embeddings_args: Optional[dict] = None

    @field_validator("name")
    def name_validator(cls, v):
        if v != VALIDATE:
            raise ValueError(f"name must be {VALIDATE}")
        return v.title()


class TrainingDTO(BaseModel):
    total_steps: int
    train_loop: list[LoopObjectDTO]
