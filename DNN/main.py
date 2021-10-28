import torchvision

import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.datamodules import FashionMNISTDataModule

from model import ConvNet, DenseNet
from utils import count_parameters
from constants import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATH_DATASETS,
    BATCH_SIZE,
    NUM_WORKERS,
    AVAIL_GPUS,
)


STD_MODELS = {
    "baseline_0": {"arch": [784], "activation_fn": None},
    # Effect of Activation Functions
    "256_sigmoid": {"arch": [784, 256], "activation_fn": "sigmoid"},
    "256_tanh": {"arch": [784, 256], "activation_fn": "tanh"},
    "256_relu": {"arch": [784, 256]},
    # Effect of Addition of Layers 1
    "200_ReLU_200_ReLU": {"arch": [784, 200, 200], "activation_fn": "relu"},
    "250_ReLU_32_ReLU": {"arch": [784, 250, 32], "activation_fn": "relu"},
    # Effect of Addition of Layers 2
    "512_relu": {"arch": [784, 512]},
    "350_ReLU_350_ReLU": {"arch": [784, 350, 350]},
    "400_ReLU_128_ReLU": {"arch": [784, 400, 128]},
    # Effect of Addition of Layers 3
    "2048_relu": {"arch": [784, 2048]},
    "920_ReLU_920_ReLU": {"arch": [784, 920, 920]},
    "720_ReLU_720_ReLU_720_ReLU": {"arch": [784, 720, 720, 720]},
    # Encoder Type Models
    "512_ReLU_128_ReLU": {"arch": [784, 512, 128]},
    "5_ReLU": {"arch": [784, 5]},
    "512_ReLU_256_ReLU_128_ReLU_64_ReLU_32_ReLU_16_ReLU": {
        "arch": [784, 256, 128, 64, 32, 16]
    },
    "256_ReLU_64_ReLU_128_ReLU": {"arch": [784, 256, 64, 128]},
    "1024_ReLU_2048_ReLU_2048_ReLU_2048_ReLU_256_ReLU_10": {
        "arch": [784, 1024, 2048, 2048, 2048, 256, 10]
    },
}


def test_model(dm, name, model):

    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=EPOCHS,
        gpus=AVAIL_GPUS,
        logger=TensorBoardLogger("test_runs/", name=name),
    )

    trainer.fit(model, datamodule=dm)

    for exp_idx in range(10):

        results = []
        (res,) = trainer.test(model, datamodule=dm)

        results.append(
            {
                "exp_idx": exp_idx,
                "name": name,
                "parameters": count_parameters(model),
                "epochs": EPOCHS,
                **res,
                "lr": LEARNING_RATE,
                "weight_decary": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
            }
        )

    return results


def main():
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
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

    results = []

    # Test Standard FC Models
    for name, layers in STD_MODELS.items():
        model = DenseNet(**layers, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        res = test_model(fashion_mnist_dm, name, model)
        results.append(res)

    # Test the Conv Net
    conv_net = ConvNet(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    res = test_model(fashion_mnist_dm, "conv_baseline", conv_net)
    results.append(res)

    results = pd.DataFrame(results)
    results.to_csv("results.csv")


if __name__ == "__main__":
    main()
