__all__ = ["WandbTracker", "get_tracker_class", "ImagePredictionLogger"]


from .callbacks import ImagePredictionLogger
from .wandb import WandbTracker


def get_tracker_class(tracker: str) -> WandbTracker:
    """
    Get tracker class by name.

    :param tracker: The name of the tracker.
    :type tracker: str
    :return: The tracker class.
    :rtype: WandbTracker
    :raises ValueError: If the tracker is not found.
    """
    if tracker == "wandb":
        return WandbTracker
    else:
        raise ValueError(f"Tracker {tracker} not found")
