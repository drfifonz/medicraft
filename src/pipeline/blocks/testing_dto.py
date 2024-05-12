from pydantic import BaseModel, Field, computed_field


class TestingDTO(BaseModel):
    train_generator: str = Field(..., description="The training generator block")

    # loop: list
