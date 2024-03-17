import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ProgressBar
from lightning.pytorch.loggers import WandbLogger

import wandb
from datasets import EyeScans
from models import ResNetClassifier
from trackers import ImagePredictionLogger


def main():
    DATASET_DIR = "datasets/genuine_eyes_ballanced"

    dm = EyeScans(data_dir=DATASET_DIR, batch_size=32)
    dm.setup()

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(dm.val_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]
    print(val_imgs.shape, val_labels.shape)
    # raise ValueError("Testing only")

    model = ResNetClassifier(num_classes=2)

    wandb_logger = WandbLogger(project="wandb-lightning-test", job_type="train")
    early_stop_callback = EarlyStopping(monitor="val_loss")
    # checkpoint_callback = ModelCheckpoint()
    progressbar_callback = ProgressBar()
    trainer = pl.Trainer(
        max_epochs=50,
        # progress_bar_refresh_rate=20,
        # gpus=1,
        logger=wandb_logger,
        callbacks=[early_stop_callback, ImagePredictionLogger(val_samples), progressbar_callback],
        # checkpoint_callback=checkpoint_callback,
        enable_checkpointing=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)
    trainer.test()

    wandb.finish()


if __name__ == "__main__":
    main()