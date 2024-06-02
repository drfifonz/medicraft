from typing import Any, Optional

from pydantic import BaseModel


class ModelsDTO(BaseModel):
    unet: Optional[dict[str, Any]] = None
    diffusion: Optional[dict[str, Any]] = None
    classifier: Optional[dict[str, Any]] = None
