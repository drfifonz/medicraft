from pydantic import BaseModel


class GeneralDTO(BaseModel):
    total_steps: int
