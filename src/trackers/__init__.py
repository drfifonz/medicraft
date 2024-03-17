__all__ = ["WandbTracker", "get_tracker_class", "ImagePredictionLogger"]


from .callbacks import ImagePredictionLogger
from .wandb import WandbTracker


def get_tracker_class(tracker: str) -> WandbTracker:  # TODO add abstract tracker class as return type
    """Get tracker class by name."""

    if tracker == "wandb":
        return WandbTracker
    else:
        raise ValueError(f"Tracker {tracker} not found")
