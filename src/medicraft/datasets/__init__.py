"""
This module provides datasets and data loading methods related to eye scans and ophthalmology.
"""

__all__ = [
    "EyeScans",
    "EyeScansV2",
    "OpthalAnonymizedDataset",
    "get_csv_dataset",
]

from .eye_scans import EyeScans
from .eye_scansv2 import EyeScansV2
from .opthal_anonymized import OpthalAnonymizedDataset, get_csv_dataset
