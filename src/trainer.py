from denoising_diffusion_pytorch import Trainer as DiffusionTrainer


class Trainer(DiffusionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.model = self.diffusion.model
        # self.model.to(device=self.device)
        # self.model.eval()

    def sample(self, batch_size: int = 1):
        return self.model.sample(batch_size=batch_size)
