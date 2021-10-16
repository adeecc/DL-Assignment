from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import pytorch_lightning as pl
from torchmetrics.functional import accuracy, precision_recall

from constants import classes
from utils import plot_classes_preds


class Net(pl.LightningModule):
    def __init__(self, layers: Dict[str, nn.Module] = None) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(layers)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        for k, layer in self.layers.items():
            x = layer(x)

        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss = F.cross_entropy(self(images), labels)

        if self.current_epoch == 1:
            sample_img = torch.rand((1, 1, 28, 28), device=self.device)
            self.logger.experiment.add_graph(Net(self.layers), sample_img)

        if batch_idx % 1000 == 0:
            self.logger.experiment.add_figure(
                "predictions vs actual",
                plot_classes_preds(self, images, labels),
                global_step=self.current_epoch,
            )

            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.current_epoch)

        self.log("train_loss", loss)
        return loss

    def _calculate_loss(self, batch, stage=None):
        x, y = batch
        out = self(x)

        loss = F.cross_entropy(out, y)
        preds = torch.argmax(out, dim=-1)

        acc = accuracy(preds, y)
        prec, rec = precision_recall(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
            self.log(f"{stage}_prec", prec, prog_bar=True)
            self.log(f"{stage}_rec", rec, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, "val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=5e-4)
        return optimizer
