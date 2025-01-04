import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import torch
from config import SPOT_CHECKPOINT_DIR
from pydantic import BaseModel
from torch import nn
from trackers.wandb import WandbTracker
from utils.classproperty import classproperty

MODEL_NAME = "checkpoint.pth"


class SpotRunStatus(BaseModel):
    current_stage: Optional[str] = ""
    current_step: int

    config: dict

    tracker_run_name: Optional[str] = None


class SpotCheckpointer:

    @staticmethod
    def remove_checkpoints(dir: Path = SPOT_CHECKPOINT_DIR) -> None:
        """
        Remove all the checkpoints.
        """
        logging.info(f"Removing all checkpoints from {dir}")
        if dir.exists() and dir.is_dir():
            shutil.rmtree(dir)
            logging.info("All checkpoints removed.")
        else:
            logging.warning("Checkpoint directory does not exist.")

    @classmethod  # TODO might be removed
    def save(
        cls,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        run_status: SpotRunStatus,
    ):
        SPOT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        current_stage = run_status.current_stage
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(checkpoint, SPOT_CHECKPOINT_DIR / current_stage / MODEL_NAME)
        cls._save_json_status(run_status)

    @classmethod
    def load(
        cls,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> None:
        run_status = cls.get_status()
        stage = run_status.current_stage

        checkpoint = torch.load(SPOT_CHECKPOINT_DIR / stage / MODEL_NAME)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    @staticmethod
    def get_model(stage: str) -> nn.Module:
        checkpoint = torch.load(SPOT_CHECKPOINT_DIR / stage / MODEL_NAME)
        return checkpoint["model"]

    @staticmethod
    def get_status() -> SpotRunStatus:
        with open(SPOT_CHECKPOINT_DIR / "status.json", "r") as f:
            return SpotRunStatus(**json.load(f))

    @classmethod
    def _save_json_status(cls, run_status: SpotRunStatus) -> None:
        # run_status.tracker_run_name = cls.__get_tracker_run_name()
        # print(run_status)
        with open(SPOT_CHECKPOINT_DIR / "status.json", "w") as f:
            json.dump(run_status.model_dump(), f, indent=4)

    @staticmethod
    def checkpoint_exists() -> bool:
        return SPOT_CHECKPOINT_DIR.exists()

    @staticmethod
    def __get_tracker_run_name(tracker: WandbTracker) -> str:  # TODO maybe to be removed
        return tracker.get_experiment_name()

    @classproperty
    def status(cls) -> SpotRunStatus:
        return cls.get_status()
