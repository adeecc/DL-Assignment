import torch
import torch.nn as nn
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.datamodules import FashionMNISTDataModule

from model import Net
from constants import PATH_DATASETS, BATCH_SIZE, NUM_WORKERS, AVAIL_GPUS


STD_MODELS = {
    "baseline_0": {
        "fc1": nn.Linear(28 * 28, 10),
    },
    "300_sigmoid": {
        "fc1": nn.Linear(28 * 28, 300),
        "ac1": nn.Sigmoid(),
        "fc2": nn.Linear(300, 10),
    },
    "300_tanh": {
        "fc1": nn.Linear(28 * 28, 300),
        "ac1": nn.Tanh(),
        "fc2": nn.Linear(300, 10),
    },
    "300_relu": {
        "fc1": nn.Linear(28 * 28, 300),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(300, 10),
    },
    "200_ReLU_200_ReLU": {
        "fc1": nn.Linear(28 * 28, 200),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(200, 200),
        "ac2": nn.ReLU(),
        "fc3": nn.Linear(200, 10),
    },
    "200_ReLU_200_ReLU_200_ReLU": {
        "fc1": nn.Linear(28 * 28, 200),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(200, 200),
        "ac2": nn.ReLU(),
        "fc3": nn.Linear(200, 200),
        "ac3": nn.ReLU(),
        "fc4": nn.Linear(200, 10),
    },
    "512_ReLU_128_ReLU_32_ReLU": {
        "fc1": nn.Linear(28 * 28, 512),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(512, 128),
        "ac2": nn.ReLU(),
        "fc3": nn.Linear(128, 32),
        "ac3": nn.ReLU(),
        "fc4": nn.Linear(32, 10),
    },
    "5_ReLU": {
        "fc1": nn.Linear(28 * 28, 5),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(5, 10),
    },
    "2048_ReLU": {
        "fc1": nn.Linear(28 * 28, 2048),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(2048, 10),
    },
}


def create_layer_dict():
    layers = {
        "fc1": nn.Linear(28 * 28, 300),
        "ac1": nn.ReLU(),
        # "ac1": nn.Sigmoid(),
        # "ac1": nn.Tanh(),
        "fc2": nn.Linear(300, 10),
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

    for name, layers in STD_MODELS.items():
        fashion_mnist_model = Net(layers)

        trainer = pl.Trainer(
            progress_bar_refresh_rate=10,
            max_epochs=30,
            gpus=AVAIL_GPUS,
            logger=TensorBoardLogger("runs/", name=name),
        )

        trainer.fit(fashion_mnist_model, datamodule=fashion_mnist_dm)
        trainer.test(fashion_mnist_model, datamodule=fashion_mnist_dm)


if __name__ == "__main__":
    main()
