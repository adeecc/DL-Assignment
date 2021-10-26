import torch
import torch.nn as nn
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.datamodules import FashionMNISTDataModule

from model import Net
from constants import PATH_DATASETS, BATCH_SIZE, NUM_WORKERS, AVAIL_GPUS


STD_MODELS = {
    # Multiclass Logistic Regression
    "baseline_0": {
        "fc1": nn.Linear(28 * 28, 10),
    },
    # Effect of Activation Functions
    "256_sigmoid": {
        "fc1": nn.Linear(28 * 28, 256),
        "ac1": nn.Sigmoid(),
        "fc2": nn.Linear(256, 10),
    },
    "256_tanh": {
        "fc1": nn.Linear(28 * 28, 256),
        "ac1": nn.Tanh(),
        "fc2": nn.Linear(256, 10),
    },
    "256_relu": {
        "fc1": nn.Linear(28 * 28, 256),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(256, 10),
    },
    # Effect of Addition of Layers
    "512_relu": {
        "fc1": nn.Linear(28 * 28, 512),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(512, 10),
    },
    "350_ReLU_350_ReLU": {
        "fc1": nn.Linear(28 * 28, 350),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(350, 350),
        "ac2": nn.ReLU(),
        "fc3": nn.Linear(350, 10),
    },
    "400_ReLU_128_ReLU": {
        "fc1": nn.Linear(28 * 28, 400),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(400, 128),
        "ac2": nn.ReLU(),
        "fc3": nn.Linear(128, 10),
    },
    # Effect of Addition of Layers
    "2048_relu": {
        "fc1": nn.Linear(28 * 28, 2048),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(2048, 10),
    },
    "920_ReLU_920_ReLU": {
        "fc1": nn.Linear(28 * 28, 920),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(920, 920),
        "ac2": nn.ReLU(),
        "fc3": nn.Linear(920, 10),
    },
    "720_ReLU_720_ReLU_720_ReLU": {
        "fc1": nn.Linear(28 * 28, 720),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(720, 720),
        "ac2": nn.ReLU(),
        "fc3": nn.Linear(720, 720),
        "ac3": nn.ReLU(),
        "fc4": nn.Linear(720, 10),
    },
    # Encoder Type Models
    "512_ReLU_128_ReLU": {
        "fc1": nn.Linear(28 * 28, 512),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(512, 128),
        "ac2": nn.ReLU(),
        "fc3": nn.Linear(128, 10),
    },
    "5_ReLU": {
        "fc1": nn.Linear(28 * 28, 5),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(5, 10),
    },
    "512_ReLU_256_ReLU_128_ReLU_64_ReLU_32_ReLU_16_ReLU": {
        "fc1": nn.Linear(28 * 28, 512),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(512, 256),
        "ac2": nn.ReLU(),
        "fc3": nn.Linear(256, 128),
        "ac3": nn.ReLU(),
        "fc4": nn.Linear(128, 64),
        "ac4": nn.ReLU(),
        "fc5": nn.Linear(64, 32),
        "ac5": nn.ReLU(),
        "fc6": nn.Linear(32, 16),
        "ac6": nn.ReLU(),
        "fc7": nn.Linear(16, 10),
    },
    "512_ReLU_128_ReLU": {
        "fc1": nn.Linear(28 * 28, 256),
        "ac1": nn.ReLU(),
        "fc2": nn.Linear(256, 64),
        "ac2": nn.ReLU(),
        "fc3": nn.Linear(64, 128),
        "ac3": nn.ReLU(),
        "fc7": nn.Linear(128, 10),
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
