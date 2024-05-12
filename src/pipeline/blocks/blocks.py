from enum import Enum

from .data_dto import DataDTO
from .general_dto import GeneralDTO
from .output_dto import OutputDTO
from .training_dto import TrainingDTO


class ConfigBlocks(Enum):
    """
    Enum for the config blocks
    """

    general = GeneralDTO
    data = DataDTO
    training = TrainingDTO
    output = OutputDTO
