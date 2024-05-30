from typing import Optional

from pydantic import BaseModel


class OutputDTO(BaseModel):
    results_dir: str = ".results"
    move_results_to: Optional[str] = None
