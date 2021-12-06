import torch
import torch.nn as nn 
import torch.nn.functional as F

import pandas as pd 
import torchvision

import pytorch_lightning as pl
from pl_bolts.datamodules import FashionMNISTDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from utils import count_parameters
import torchvision.models as models 
from torchmetrics.functional import accuracy, precision, recall

from constants import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATH_DATASETS,
    BATCH_SIZE,
    NUM_WORKERS,
    AVAIL_GPUS,
)

class ResNet(pl.LightningModule):

    def __init__(
        self, 
        in_channels: int = 1,
        out_channels: int = 64,
        kernel_size: int = 7,
        stride: int = 2, 
        padding: int = 3, 
        num_classes:int = 10,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        loss_fn: str = "cross_entropy"
    ) -> None: 

        super().__init__()
        self.save_hyperparameters("lr", "weight_decay", "loss_fn")
        self.num_classes = num_classes

        self.model = models.resnet50(pretrained=True)

        # RGB -> Grayscale => in_channels = 1

        self.model.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, bias=False
        )

        # Output layer 10 classes 
        num_features = self.model.fc.in_features 
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch 
        outs = self(images)

        loss = F.cross_entropy(outs, labels)

        self.log("train_loss", loss)
        return loss 
    
    def evaluate(self, batch, stage=None):
        images, labels = batch
        outs = self(images)


        loss = F.cross_entropy(outs, labels)

        acc = accuracy(outs, labels, num_classes=self.num_classes)
        prec = precision(outs, labels, num_classes=self.num_classes, average="macro")
        rec = recall(outs, labels, num_classes=self.num_classes, average="macro")

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
            lr = self.hparams.lr, 
            weight_decay=self.hparams.weight_decay 
        )

        return optimizer

def test_model(dm, name, model):

    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=EPOCHS,
        gpus=AVAIL_GPUS,
        logger=TensorBoardLogger("resnet_runs/", name=name),
    )

    print(f"Starting Training for {name}")
    trainer.fit(model, datamodule=dm)

    (res,) = trainer.test(model, datamodule=dm)

    return {
        "name": name,
        "parameters": count_parameters(model),
        "epochs": EPOCHS,
        **res,
        "lr": LEARNING_RATE,
        "weight_decary": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
    }

    
if __name__ == "__main___":
    seed_everything(42)

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    fashion_mnist_dm = FashionMNISTDataModule(
        data_dir=PATH_DATASETS,
        batch_size=BATCH_SIZE,
        normalize=True,
        num_workers=NUM_WORKERS,
        train_transforms=transforms,
        val_transforms=transforms,
        test_transforms=transforms,
    )

    results = []
    res_net = ResNet(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    res = test_model(fashion_mnist_dm, "resnet_50", res_net)
    results.append(res)

    results = pd.DataFrame(results)
    results.to_csv("resnet_results.csv")