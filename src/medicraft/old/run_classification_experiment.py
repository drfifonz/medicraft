import argparse

import lightning as pl
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

import wandb
from datasets import EyeScans
from models import ResNetClassifier
from trackers import ImagePredictionLogger

parser = argparse.ArgumentParser(description="Run a classification experiment.")
parser.add_argument("--real", action="store_true", help="Use real-world data.")


def main():
    TRAIN_DATASET_DIR = "data/datasets/ophthal_anonym_classed-stratified/train"
    VAL_DATASET_DIR = "data/datasets/ophthal_anonym_classed-stratified/val"
    TEST_DATASET_DIR = "data/datasets/ophthal_anonym_classed-stratified/test"

    SYNTH_TRAIN_DATASET_DIR = "data/datasets/synthetic/0001"

    args = parser.parse_args()
    if args.real:
        REAL_WORD_DATA = True
    else:
        REAL_WORD_DATA = False

    train_data_dir, val_data_dir = (
        (TRAIN_DATASET_DIR, VAL_DATASET_DIR) if REAL_WORD_DATA else (SYNTH_TRAIN_DATASET_DIR, None)
    )

    dm = EyeScans(
        num_workers=4,
        batch_size=32,
        ratio=[0.8, 0.2],
        real_word_data=REAL_WORD_DATA,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        test_dataset_dir=TEST_DATASET_DIR,
    )
    dm.setup()

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(dm.val_dataloader()))
    # val_imgs, val_labels = val_samples[0], val_samples[1]
    # print(val_imgs.shape, val_labels.shape)

    # raise

    loss_multiply = 1.0

    model = ResNetClassifier(
        num_classes=4,
        loss_fn=nn.CrossEntropyLoss(),
        pretrained=False,
        learning_rate=1e-3,
        loss_multiply=loss_multiply,
    )

    dataset_type = "real" if REAL_WORD_DATA else "synthetic"
    wandb_logger = WandbLogger(
        project="medicraft-classification",
        # mode="offline",
        job_type="train",
        tags=[
            "resnet34",
            # "pretrained",
            dataset_type,
            "center-crop",
            "average-macro",
            "trained-on-straified",
            "tested-on-straified",
        ],
    )
    early_stop_callback = EarlyStopping(monitor="val_loss")
    # checkpoint_callback = ModelCheckpoint()
    progressbar_callback = TQDMProgressBar()
    trainer = pl.Trainer(
        max_epochs=15,
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
    trainer.test(model, dm)

    wandb.finish()


if __name__ == "__main__":
    main()
