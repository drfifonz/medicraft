from enum import Enum

from .data_dto import DataDTO
from .experiment_dto import ExperimentDTO
from .general_dto import GeneralDTO
from .output_dto import OutputDTO


class ConfigBlocks(Enum):
    """
    Enum for the config blocks
    """

    general = GeneralDTO
    data = DataDTO
    experiment = ExperimentDTO
    output = OutputDTO
