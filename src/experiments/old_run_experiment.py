import torch
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=1,
)
model.to(device=DEVICE)

diffusion = GaussianDiffusion(
    model,
    image_size=(512, 256)[::-1],
    # timesteps=1000,  # number of steps
    timesteps=2,  # number of steps
    # loss_type = 'l1'    # L1 or L2
)
diffusion.to(device=DEVICE)

print(f"Model loaded to {DEVICE} device.")
# data_path = "./data/healthy_eyes_1024x512"
data_path = "./data/healthy_eyes_256x128"

# data file path
trainer = Trainer(
    diffusion,
    data_path,
    train_batch_size=4,
    train_lr=2e-4,
    save_and_sample_every=2000,
    train_num_steps=200_000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=True,  # turn on mixed precision
    num_samples=9,  # number of samples to save
    calculate_fid=False,  # calculate FID during sampling
)
# trainer.load("healthy_eyes_512x256_s80000/model-40.pt")
trainer.train()
