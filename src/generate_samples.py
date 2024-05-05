import torch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import num_to_groups
from ema_pytorch import EMA
from torchvision import utils
from tqdm import tqdm

from models import GaussianDiffusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_samples(
    diffusion_model: GaussianDiffusion,
    results_dir: str,
    # ema_model=,
    ema_decay=0.995,
    ema_update_every=10,
    num_samples: int = 100,
    batch_size: int = 1,
    start_sample_idx: int = 0,
):
    # noise = torch.randn([16, 1, 256, 512], device=DEVICE)

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
        # print(batches)
        pbar = tqdm(total=num_samples)
        for idx, batch_size in enumerate(batches):
            # batch = batches[idx]
            images: list[torch.Tensor] = ema.ema_model.sample(batch_size=batch_size)
            for i in range(len(images)):
                # detach image from tensor to pil image

                # numpy_array = images[i].add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                # pil_image = Image.fromarray(numpy_array)
                progress = i + idx * batch_size
                utils.save_image(images[i], f"{results_dir}/image_{(start_sample_idx + progress):>06}.png")
                # save_image(pil_image, f"{results_dir}/image_{progress}.png")
                pbar.update(progress)
    pbar.close()
