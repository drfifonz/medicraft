from typing import Optional

from pipeline.blocks.models_dto import ModelsDTO
from pydantic import BaseModel


class GeneralDTO(BaseModel):
    total_steps: int = 0
    image_size: list[int]
    experiment_id: Optional[str] = None

    spot_checkpointing: bool = False
    models: ModelsDTO
