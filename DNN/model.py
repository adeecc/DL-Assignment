from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from torchmetrics.functional import accuracy, precision, recall

from utils import plot_classes_preds


class Net(pl.LightningModule):
    def __init__(self, lr: float = 3e-4, weight_decay: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss = F.cross_entropy(self(images), labels)

        if batch_idx == 0:
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
        if stage == "test":
            self.log(f"hp_metric", acc)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=3e-4, momentum=1e-3, weight_decay=1e-3)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer


class DenseNet(Net):
    def __init__(
        self,
        arch: List[int] = None,
        activation_fn: str = "relu",
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
    ) -> None:
        super().__init__(lr, weight_decay)

        layers = [nn.Flatten()]

        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i + 1]))

            if activation_fn == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation_fn == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())


        layers.append(nn.Linear(arch[-1], 10))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvNet(Net):
    def __init__(self, lr: float = 3e-4, weight_decay: float = 1e-3) -> None:
        super().__init__(lr, weight_decay)

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
