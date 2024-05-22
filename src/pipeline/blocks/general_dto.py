from pydantic import BaseModel, field_validator

from pipeline.blocks.models_dto import ModelsDTO


class GeneralDTO(BaseModel):
    total_steps: int
    image_size: list[int]

    models: ModelsDTO
