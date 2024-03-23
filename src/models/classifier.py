import lightning as pl
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchmetrics import Accuracy, F1Score


class ResNetClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        loss_fn=None,
        optimizer=None,
        pretrained: bool = True,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        self.model = models.resnet34(pretrained=pretrained)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)

        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.loss_fn = loss_fn if loss_fn else F.nll_loss
        self.optimizer = optimizer if optimizer else torch.optim.Adam

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1score = F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)
        self.log("train_f1", self.f1score(preds, y), on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # validation metricKs
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("val_loss", losson_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.f1score(preds, y), on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer
