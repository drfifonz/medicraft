from pathlib import Path

import torch
from denoising_diffusion_pytorch import Unet
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import num_to_groups
from ema_pytorch import EMA
from PIL import Image
from torchvision import utils
from tqdm import tqdm

from models import GaussianDiffusion
from utils import copy_results_directory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=1)
model.to(device=DEVICE)

diffusion = GaussianDiffusion(
    model,
    image_size=(512, 256)[::-1],
    timesteps=1000,  # number of steps
    # timesteps=2,  # number of steps
    # loss_type = 'l1'    # L1 or L2
)
diffusion.to(device=DEVICE)


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
        t = tqdm(total=num_samples)
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
                t.update(progress)
    t.close()


@torch.no_grad()
def p_sample_loop(
    self,
    model,
    noise,
    device,
    noise_fn=torch.randn,
    capture_every=1000,
):
    img = noise
    imgs = []

    for i in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps):
        img = self.p_sample(
            model,
            img,
            torch.full((img.shape[0],), i, dtype=torch.int64).to(device),
            noise_fn=noise_fn,
        )

        if i % capture_every == 0:
            imgs.append(img)

    imgs.append(img)

    return imgs


def save_image(image: Image, path) -> None:
    with open(path, "wb") as f:
        image.save(f, format="png")


if __name__ == "__main__":
    MODEL_PATH = ".results/honest-frost-6/model-150.pt"

    diffusion.load_state_dict(torch.load(MODEL_PATH)["model"])
    checkpoint = torch.load(MODEL_PATH)  # TODO unused variable ?

    DATASET_PATH = "datasets/with_fluid_eyes_512x256_dataset"
    Path.mkdir(Path(DATASET_PATH), parents=True, exist_ok=True)
    ONEDRIVE_DATASETS_DIR = "/home/wmi/OneDrive/General/results/datasets/"
    ONEDRIVE_DESTINATION_DIR = str(Path(ONEDRIVE_DATASETS_DIR) / Path(DATASET_PATH).name)

    generate_samples(
        diffusion,
        DATASET_PATH,
        num_samples=2000,
        batch_size=10,
        # start_sample_idx=2010,
    )

    copy_results_directory(DATASET_PATH, ONEDRIVE_DESTINATION_DIR)
