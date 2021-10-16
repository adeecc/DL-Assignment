import torch
import torch.nn as nn
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.datamodules import FashionMNISTDataModule

from model import Net
from constants import PATH_DATASETS, BATCH_SIZE, NUM_WORKERS, AVAIL_GPUS


def layer_dict():
    layers = {
        "fc1": nn.Linear(28 * 28, 300),
        "bn1": nn.BatchNorm1d(300),
        "relu1": nn.ReLU(),
        "droput1": nn.Dropout(0.1),
        "fc2": nn.Linear(300, 100),
        "bn2": nn.BatchNorm1d(100),
        "relu2": nn.ReLU(),
        "dropout2": nn.Dropout(0.1),
        "fc3": nn.Linear(100, 10),
    }

    return layers


def main():
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    fashion_mnist_dm = FashionMNISTDataModule(
        data_dir=PATH_DATASETS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_transforms=transforms,
        val_transforms=transforms,
        test_transforms=transforms,
    )

    fashion_mnist_model = Net(layer_dict())

    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=30,
        gpus=AVAIL_GPUS,
        logger=TensorBoardLogger("runs/", name="test_runs"),
    )

    trainer.fit(fashion_mnist_model, datamodule=fashion_mnist_dm)
    trainer.test(fashion_mnist_model, datamodule=fashion_mnist_dm)


if __name__ == "__main__":
    main()
