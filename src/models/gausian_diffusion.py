from denoising_diffusion_pytorch import GaussianDiffusion as GausianDiffusionModel
import torch


class GaussianDiffusion(GausianDiffusionModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
