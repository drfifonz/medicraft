import torch
from denoising_diffusion_pytorch import Unet
from torchvision import transforms as T

import config as cfg
from datasets import OpthalAnonymizedDataset, get_csv_dataset
from models import GaussianDiffusion
from trainers import Trainer
from utils import copy_results_directory
from utils.transforms import HorizontalCenterCrop

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = (256, 512)


def main():

    df = get_csv_dataset(cfg.DATASET_FILE_PATH)["train"]

    DIAGNOSIS = "reference"  # ["precancerous", "fluid", "benign", "reference"]
    transform = T.Compose(
        [
            HorizontalCenterCrop(512),
            T.Resize(IMAGE_SIZE),
            T.RandomHorizontalFlip(),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
        ]
    )
    dataset = OpthalAnonymizedDataset(
        diagnosis=DIAGNOSIS,
        df=df,
        images_dir=cfg.DATASET_FILE_PATH.parent / "images",
        transform=transform,
        convert_image_to="L",
    )

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1,
    )

    model.to(device=DEVICE)

    diffusion = GaussianDiffusion(
        model,
        image_size=IMAGE_SIZE,
        timesteps=1000,  # number of steps #
        # timesteps=2,  # number of steps
        # loss_type = 'l1'    # L1 or L2
    )
    diffusion.to(device=DEVICE)
    print(f"Model loaded to {DEVICE} device.")

    # data_path = "./data/input_datasets/with_fluid_eyes_512x256"
    # data_path = "./data/input_datasets/healthy_eyes_512x256"

    experiment_id = "0003"
    trainer = Trainer(  # noqa : F841
        diffusion,
        str(cfg.DATASET_FILE_PATH.parent / "images"),
        dataset=dataset,
        train_batch_size=4,
        train_lr=2e-4,
        save_and_sample_every=2000,
        # save_and_sample_every=10,
        results_folder=f"./.results/{experiment_id}/{DIAGNOSIS}",
        train_num_steps=100_000,  # total training steps
        gradient_accumulate_every=4,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        num_samples=9,  # number of samples to save
        calculate_fid=False,  # calculate FID during sampling
        tracker="wandb",
        tracker_kwargs={
            "tags": [DIAGNOSIS, "opthal_anonymized"],
            "mode": "online",
        },
    )
    # trainer.load(".results/0002/reference/model-30.pt")
    trainer.train()
    copy_results_directory(
        f"./.results/{experiment_id}/{DIAGNOSIS}",
        f"/home/wmi/OneDrive/General/results/05.2024/.results/{experiment_id}/",
    )


if __name__ == "__main__":
    main()
