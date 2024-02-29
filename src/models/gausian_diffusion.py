import torch
from denoising_diffusion_pytorch import GaussianDiffusion as GausianDiffusionModel


class GaussianDiffusion(GausianDiffusionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # TODo add direct arguments passing with typing

    def forward(self, img, *args, **kwargs):
        (b, _, h, w, device, img_size) = (
            *img.shape,
            img.device,
            self.image_size,
        )
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        assert h == img_size[0] and w == img_size[1], f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

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

    def get_hyperparameters(self) -> dict:  # TODO get guassian diffusion hyperparameters
        return {}
