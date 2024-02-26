import json
import math
import os
from typing import Literal

import torch
from denoising_diffusion_pytorch import Trainer as DiffusionTrainer
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    divisible_by,
    exists,
    num_to_groups,
)
from denoising_diffusion_pytorch.version import __version__
from torch import nn
from torchvision import transforms as utils  # TODO change to other import name
from tqdm.auto import tqdm

from trackers import WandbTracker, get_tracker_class


class Trainer(DiffusionTrainer):
    def __init__(
        self,
        diffusion_model: nn.Module,
        folder: str,
        *,
        train_batch_size: int = 16,
        gradient_accumulate_every: int = 1,
        augment_horizontal_flip: int = True,
        train_lr: float = 1e-4,
        train_num_steps: int = 100000,
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
        if tracker:
            tracker_class = get_tracker_class(tracker.lower())
            self.tracker = tracker_class(
                project_name=getattr(tracker_kwargs, "project_name", "medicraft-diffusion"),
                hyperparameters={k: v for k, v in locals().items() if k != "self"},
            )

        # raise ValueError("Tracker testing only") # TODO remove this line
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

        print("Trainer initialized")

    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timesteps=False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if isinstance(image_size, tuple):
            return sample_fn(
                (batch_size, channels, *image_size),
                return_all_timesteps=return_all_timesteps,
            )

        return sample_fn(
            (batch_size, channels, image_size, image_size),
            return_all_timesteps=return_all_timesteps,
        )

    def save(self, milestone):
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

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))
        self.keep_last_models(10)

    def keep_last_models(self, num_models: int = 10) -> None:
        # keep only last num_models models
        list_of_models = sorted(self.results_folder.glob("*.pt"), key=os.path.getmtime)
        models_to_remove = list_of_models[:-num_models]
        for model in models_to_remove:
            model.unlink()

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

                pbar.update(1)

        accelerator.print("training complete")
