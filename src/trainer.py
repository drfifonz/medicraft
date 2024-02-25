import os
from denoising_diffusion_pytorch import Trainer as DiffusionTrainer, exists
from denoising_diffusion_pytorch.version import __version__

import torch


class Trainer(DiffusionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
