from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from torchmetrics.functional import accuracy, precision, recall

from utils import plot_classes_preds


class Net(pl.LightningModule):
    def __init__(
        self, layers_dict: Dict[str, nn.Module] = None, flatten: bool = True
    ) -> None:
        super().__init__()
        self.layers_dict = layers_dict
        self.flatten = flatten
        self.layers = nn.ModuleDict(self.layers_dict)

    def forward(self, x):
        if self.flatten:
            x = x.reshape(x.size(0), -1)
        for _, layer in self.layers.items():
            x = layer(x)

        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss = F.cross_entropy(self(images), labels)

        if self.current_epoch == 1 and batch_idx == 0:
            sample_img = images[0].unsqueeze(0)
            self.logger.experiment.add_graph(Net(self.layers_dict), sample_img)

        if batch_idx == 0:
            self.logger.experiment.add_figure(
                "predictions vs actual",
                plot_classes_preds(self, images, labels),
                global_step=self.current_epoch,
            )

            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.current_epoch)

        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        out = self(x)

        loss = F.cross_entropy(out, y)

        acc = accuracy(out, y, num_classes=10)
        prec = precision(out, y, num_classes=10, average="macro")
        rec = recall(out, y, num_classes=10, average="macro")

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
            self.log(f"{stage}_prec", prec)
            self.log(f"{stage}_rec", rec)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=5e-4)
        return optimizer
