from typing import Optional

from pydantic import BaseModel

from pipeline.blocks.models_dto import ModelsDTO


class GeneralDTO(BaseModel):
    total_steps: int = 0
    image_size: list[int]
    experiment_id: Optional[str] = None

    models: ModelsDTO
