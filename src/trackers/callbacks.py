import torch
import wandb
from lightning import Callback


class ImagePredictionLogger(Callback):
    """
    Callback for logging image predictions during validation.

    Args:
        val_samples (tuple): A tuple containing validation images and labels.
        num_samples (int): Number of samples to log. Default is 32.
    """

    def __init__(self, val_samples: tuple, num_samples: int = 32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Method called at the end of each validation epoch.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer object.
            pl_module (LightningModule): The PyTorch Lightning module being trained.
        """
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)

        # Log the images as wandb Image
        trainer.logger.experiment.log(
            {
                "examples": [
                    wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                    for x, pred, y in zip(
                        val_imgs[: self.num_samples], preds[: self.num_samples], val_labels[: self.num_samples]
                    )
                ]
            }
        )
