import torch
from denoising_diffusion_pytorch import Unet

from models import GaussianDiffusion
from trainers import Trainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1,
    )

    model.to(device=DEVICE)

    diffusion = GaussianDiffusion(
        model,
        image_size=(512, 256)[::-1],
        timesteps=1000,  # number of steps #
        # timesteps=2,  # number of steps
        # loss_type = 'l1'    # L1 or L2
    )
    diffusion.to(device=DEVICE)
    print(f"Model loaded to {DEVICE} device.")

    # data_path = "./data/input_datasets/with_fluid_eyes_512x256"
    data_path = "./data/input_datasets/healthy_eyes_512x256"

    trainer = Trainer(  # noqa : F841
        diffusion,
        data_path,
        train_batch_size=4,
        train_lr=2e-4,
        save_and_sample_every=2000,
        # save_and_sample_every=10,
        results_folder="./.results/continue",
        train_num_steps=150_000,  # total training steps
        gradient_accumulate_every=4,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        num_samples=9,  # number of samples to save
        calculate_fid=False,  # calculate FID during sampling
        tracker="wandb",
        tracker_kwargs={"tags": ["healthy_eyes"], "resume": True, "id": "crisp-totem-2"},
    )
    trainer.load(".results/crisp-totem-2/model-50.pt")
    trainer.train()


if __name__ == "__main__":
    main()
