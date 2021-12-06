from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from torchmetrics.functional import accuracy, precision, recall

from utils import init_weights


class Net(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        loss_fn: str = "cross_entropy",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.save_hyperparameters("lr", "weight_decay", "loss_fn")

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        out = self(images)

        if self.loss_fn == "kl_div":
            pred_probs = F.log_softmax(out, dim=-1)
            tgts = F.one_hot(labels, num_classes=self.num_classes).to(torch.float32)
            loss = F.kl_div(pred_probs, tgts, reduction="batchmean")

        else:
            loss = F.cross_entropy(out, labels)

        # if batch_idx == 0:
        #     for name, params in self.named_parameters():
        #         self.logger.experiment.add_histogram(name, params, self.current_epoch)

        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        out = self(x)

        if self.loss_fn == "kl_div":
            pred_probs = F.log_softmax(out, dim=-1)
            tgts = F.one_hot(y, num_classes=self.num_classes).to(torch.float32)
            loss = F.kl_div(pred_probs, tgts, reduction="batchmean")

        else:
            loss = F.cross_entropy(out, y)

        acc = accuracy(out, y, num_classes=self.num_classes)
        prec = precision(out, y, num_classes=self.num_classes, average="macro")
        rec = recall(out, y, num_classes=self.num_classes, average="macro")

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
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer


class DenseNet(Net):
    def __init__(
        self,
        input_size: int = 784,
        num_classes: int = 10,
        hidden_sizes: List[int] = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        activation_fn: str = "hard_tanh",
        loss_fn: str = "cross_entropy",
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, loss_fn=loss_fn)

        self.input_size = input_size
        self.num_classes = num_classes

        hidden_sizes = [self.input_size] + hidden_sizes
        layers = [nn.Flatten()]

        for i in range(len(hidden_sizes) - 1):
            layers += [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])]

            if activation_fn == "sigmoid":
                layers += [nn.Sigmoid()]
            elif activation_fn == "tanh":
                layers += [nn.Tanh()]
            elif activation_fn == "relu":
                layers += [nn.ReLU()]
            elif activation_fn == "mish":
                layers += [nn.Mish()]
            elif activation_fn == "leaky_relu":
                layers += [nn.LeakyReLU()]
            else:
                layers += [nn.Hardtanh()]

        layers += [nn.Linear(hidden_sizes[-1], num_classes)]

        self.net = nn.Sequential(*layers)
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)


class ConvNet(Net):
    def __init__(
        self,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        loss_fn: str = "cross_entropy",
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, loss_fn=loss_fn)

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



# VGG - 16 
VGG16 = [64, 64, 'M', 128, 128, 'M' , 256, 256, 256, 'M', ] #512, 512, 512, 'M', 512, 512, 512, 'M']
class VGG(Net):
    def __init__(
        self,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        loss_fn: str = "cross_entropy",
        in_channels: int = 1,
        num_classes: int = 10,
        architecture: List[int] = VGG16
    ) -> None:

        super().__init__(lr=lr, weight_decay=weight_decay, loss_fn=loss_fn)
        self.in_channels = in_channels

        self.net = self.create_conv_layers(architecture)
        self.fc = nn.Sequential(
            nn.Linear(784, 128),            # Check
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )


    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), 
                    nn.BatchNorm2d(x),    # Optional not in VGG16 Architecture
                    nn.ReLU()
                ]

                in_channels = x

            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

    def forward(self, x):
        outs = self.net(x)
        x = x.reshape(x.size(0), -1)
        outs = self.fc(x)

        return outs 
    

