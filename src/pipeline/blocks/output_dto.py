from typing import Optional

from pydantic import BaseModel, Field, computed_field


class OutputDTO(BaseModel):
    # train_generator: str = Field(..., description="The training generator block")

    results_dir: str = ".results"
    move_results_to: Optional[str] = None
