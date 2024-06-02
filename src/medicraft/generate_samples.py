from pathlib import Path

import torch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import num_to_groups
from ema_pytorch import EMA
from torchvision import utils
from tqdm import tqdm

from config import DEVICE
from models import GaussianDiffusion


def generate_samples(
    diffusion_model: GaussianDiffusion,
    results_dir: str,
    ema_decay=0.995,
    ema_update_every=10,
    num_samples: int = 100,
    batch_size: int = 1,
    start_sample_idx: int = 0,
):
    """
    Generate samples using the given diffusion model and save them to the specified directory.

    Args:
        diffusion_model (GaussianDiffusion): The diffusion model used for generating samples.
        results_dir (str): The directory where the generated samples will be saved.
        ema_decay (float, optional): The decay rate for the exponential moving average (EMA) of the diffusion model.
            Defaults to 0.995.
        ema_update_every (int, optional): The number of steps between EMA updates. Defaults to 10.
        num_samples (int, optional): The total number of samples to generate. Defaults to 100.
        batch_size (int, optional): The batch size used for generating samples. Defaults to 1.
        start_sample_idx (int, optional): The starting index for the generated sample filenames. Defaults to 0.
    """

    if not Path(results_dir).exists():
        Path(results_dir).mkdir(parents=True)

    ema = EMA(
        diffusion_model,
        beta=ema_decay,
        update_every=ema_update_every,
    )
    ema.to(DEVICE)

    with torch.inference_mode():
        batches = num_to_groups(
            num_samples,
            batch_size,
        )
        t = tqdm(total=num_samples)
        for idx, batch_size in enumerate(batches):
            images: list[torch.Tensor] = ema.ema_model.sample(batch_size=batch_size)
            for i in range(len(images)):
                progress = i + idx * batch_size
                utils.save_image(images[i], f"{results_dir}/image_{(start_sample_idx + progress):>06}.png")
                t.update(progress)
    t.close()
