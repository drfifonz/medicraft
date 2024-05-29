import math
import os
from pathlib import Path
from typing import Literal

import torch
from denoising_diffusion_pytorch import Trainer as DiffusionTrainer
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Dataset as _Dataset
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    cpu_count,
    cycle,
    divisible_by,
    exists,
    num_to_groups,
)
from denoising_diffusion_pytorch.version import __version__
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import utils  # TODO change to other import name
from tqdm.auto import tqdm

from trackers import get_tracker_class
from trackers.wandb import WandbTracker


class Trainer(DiffusionTrainer):
    def __init__(
        self,
        diffusion_model: nn.Module,
        folder: str = None,
        *,
        dataset: Dataset = None,
        train_batch_size: int = 16,
        gradient_accumulate_every: int = 1,
        augment_horizontal_flip: int = True,
        train_lr: float = 1e-4,
        train_num_steps: int = 100_000,
        break_every_steps: int | None = None,
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        adam_betas: tuple[float, float] = (0.9, 0.99),
        save_and_sample_every: int = 1000,
        num_samples: int = 25,
        results_folder: str = "./results",
        amp: bool = False,
        mixed_precision_type: str = "fp16",
        split_batches: bool = True,
        convert_image_to: Literal["L", "RGB", "RGBA"] | None = None,
        calculate_fid: bool = True,
        inception_block_idx: int = 2048,
        max_grad_norm: float = 1.0,
        num_fid_samples: int = 50000,
        save_best_and_latest_only: bool = False,
        tracker: str | None = None,
        tracker_kwargs: dict | None = None,
    ):
        self.tracker = None  # TODO try if is it necessary or remove this line
        if tracker:
            parameters = {k: v for k, v in locals().items() if k != "self"}
            [
                parameters.pop(key)  # TODO pop more keys
                for key in [
                    "tracker_kwargs",
                    "results_folder",
                    "convert_image_to",
                    "save_and_sample_every",
                    # "diffusion_model",
                    "folder",
                    "tracker",
                ]
            ]
            parameters["dataset_name"] = Path(folder).name if folder else dataset.diagnosis

            # print(tracker_kwargs)
            # group = tracker_kwargs.get("group", "diffusion")
            # raise
            # tracker_class = get_tracker_class(tracker.lower()) #TODO fix tracker selection
            self.tracker = WandbTracker(
                project_name=tracker_kwargs.get("project_name", "medicraft"),
                hyperparameters=parameters,
                tags=tracker_kwargs.get("tags", None),
                # tags=getattr(tracker_kwargs, "tags", None),
                group=tracker_kwargs.get("group", "diffusion"),
                resume=tracker_kwargs.get("resume", None),
                id=tracker_kwargs.get("id", None),
                mode=tracker_kwargs.get("mode", "online"),
            )

        # raise ValueError("Tracker testing only") # TODO remove this line

        self.break_every_steps = None

        Path(results_folder).mkdir(parents=True, exist_ok=True)
        super().__init__(
            diffusion_model=diffusion_model,
            folder=folder,
            train_batch_size=train_batch_size,
            gradient_accumulate_every=gradient_accumulate_every,
            augment_horizontal_flip=augment_horizontal_flip,
            train_lr=train_lr,
            train_num_steps=train_num_steps,
            ema_update_every=ema_update_every,
            ema_decay=ema_decay,
            adam_betas=adam_betas,
            save_and_sample_every=save_and_sample_every,
            num_samples=num_samples,
            results_folder=results_folder,
            amp=amp,
            mixed_precision_type=mixed_precision_type,
            split_batches=split_batches,
            convert_image_to=convert_image_to,
            calculate_fid=calculate_fid,
            inception_block_idx=inception_block_idx,
            max_grad_norm=max_grad_norm,
            num_fid_samples=num_fid_samples,
            save_best_and_latest_only=save_best_and_latest_only,
        )
        if dataset is None:
            self.ds = _Dataset(
                folder,
                self.image_size,
                augment_horizontal_flip=augment_horizontal_flip,
                convert_image_to=convert_image_to,
            )
        else:
            self.ds = dataset
        print("Dataset initialized")
        assert (
            len(self.ds) >= 100
        ), "you should have at least 100 images in your folder. at least 10k images recommended"

        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        print("Trainer initialized")

    def save(self, milestone: str, keep_last_models: int | None = 10):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            "version": __version__,
        }
        model_path = str(self.results_folder / f"model-{milestone}.pt")
        torch.save(data, model_path)
        torch.save(data, str(self.results_folder / "latest.pt"))
        self.keep_last_models(keep_last_models)

    def keep_last_models(self, num_models: int = 10) -> None:
        # keep only last num_models models
        list_of_models = sorted(self.results_folder.glob("*.pt"), key=os.path.getmtime)
        models_to_remove = list_of_models[:-num_models]
        for model in models_to_remove:
            if model.stem == "latest":
                continue
            model.unlink()

    def load(self, milestone_path: str):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(milestone_path, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if "version" in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f"loss: {total_loss:.4f}")
                if self.tracker:
                    self.tracker.log({"loss": total_loss})

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim=0)

                        if self.tracker:
                            self.tracker.log_images(all_images)

                        utils.save_image(
                            all_images,
                            str(self.results_folder / f"sample-{milestone}.png"),
                            nrow=int(math.sqrt(self.num_samples)),
                        )

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f"fid_score: {fid_score}")
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)
                        if self.break_every_steps and self.step % self.break_every_steps == 0:
                            self.save(f"{milestone}-val")

                pbar.update(1)
                if self.tracker:
                    self.tracker.update_step()

        accelerator.print("training complete")
