from typing import Literal

import lightning as pl
import torch
import torch.nn.functional as F
import torchvision.models as models
import wandb
from torchmetrics import Accuracy, F1Score, Precision, Recall


class ResNetClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        loss_fn=None,
        optimizer=None,
        pretrained: bool = True,
        architecture: Literal["resnet18", "resnet34", "resnet50"] = "resnet34",
        learning_rate: float = 1e-4,
        loss_multiply: float = 1.0,
        class_names: list[str] = ["fluid", "benign", "precancerous", "reference"],
    ) -> None:
        """
        Initializes the Classifier model used in validation process.

        Args:
            num_classes (int): The number of classes for classification.
            loss_fn (callable, optional): The loss function to use. Defaults to None.
            optimizer (callable, optional): The optimizer to use. Defaults to None.
            pretrained (bool, optional): Whether to use a pretrained model. Defaults to True.
            architecture (Literal["resnet18", "resnet34", "resnet50"], optional): The architecture of the model. Defaults to "resnet34".
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-4.
            loss_multiply (float, optional): The loss multiplier. Defaults to 1.0.
            class_names (list[str], optional): The names of the classes. Defaults to ["fluid", "benign", "precancerous", "reference"].
        """

        self.class_names = class_names
        super().__init__()
        self.model = self.__get_model(architecture, pretrained)
        fc_output = self.model.fc.out_features

        self.head = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(fc_output, num_classes),
        )

        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.loss_fn = loss_fn if loss_fn else F.nll_loss
        self.optimizer = optimizer if optimizer else torch.optim.Adam

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.f1score = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro")

        self.loss_multiply = loss_multiply

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y) * self.loss_multiply

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1score(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)
        self.log("train_f1", f1, on_step=True, on_epoch=True, logger=True)
        self.log("train_precision", precision, on_step=True, on_epoch=True, logger=True)
        self.log("train_recall", recall, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y) * self.loss_multiply

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1score(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)
        self.log("val_precision", precision, on_epoch=True, prog_bar=True)
        self.log("val_recall", recall, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.append({"y": y.cpu(), "y_hat": preds.cpu()})
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        y_true = torch.cat([output["y"] for output in outputs]).tolist()
        y_pred = torch.cat([output["y_hat"] for output in outputs]).tolist()

        wandb.log(
            {
                "confusion_mat": wandb.plot.confusion_matrix(
                    probs=None, y_true=y_true, preds=y_pred, class_names=self.class_names
                )
            }
        )
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y) * self.loss_multiply

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        f1 = self.f1score(preds, y)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_recall", recall, prog_bar=True)

        self.test_step_outputs.append({"y": y.cpu(), "y_hat": preds.cpu()})
        return loss

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        y_true = torch.cat([output["y"] for output in outputs]).tolist()
        y_pred = torch.cat([output["y_hat"] for output in outputs]).tolist()

        wandb.log(
            {
                "confusion_mat": wandb.plot.confusion_matrix(
                    probs=None, y_true=y_true, preds=y_pred, class_names=self.class_names
                )
            }
        )
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer

    def __get_model(self, architecture: str, pretrained: bool) -> models.ResNet:
        match architecture:
            case "resnet18":
                model = models.resnet18(pretrained=pretrained)
            case "resnet34":
                model = models.resnet34(pretrained=pretrained)
            case "resnet50":
                model = models.resnet50(pretrained=pretrained)
            case _:
                raise ValueError(f"Invalid architecture: {architecture}")
        return model
