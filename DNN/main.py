import torchvision

import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pl_bolts.datamodules import FashionMNISTDataModule

from model import ConvNet, DenseNet
from pretrained_resnet import ResNet

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
    "baseline_0": {"hidden_sizes": []},
    # Effect of Activation Functions
    "128_sigmoid": {"hidden_sizes": [128], "activation_fn": "sigmoid"},
    "128_tanh": {"hidden_sizes": [128], "activation_fn": "tanh"},
    "128_hardtanh": {"hidden_sizes": [128], "activation_fn": "hard_tanh"},
    "128_mish": {"hidden_sizes": [128], "activation_fn": "mish"},
    "128_leakyrelu": {"hidden_sizes": [128], "activation_fn": "leaky_relu"},
    "128_relu": {"hidden_sizes": [128], "activation_fn": "relu"},
    # Effect of Addition of Layers 1
    "115_ReLU_115_ReLU": {"hidden_sizes": [115, 115]},
    "125_ReLU_27_ReLU": {"hidden_sizes": [125, 27]},
    # Effect of Addition of Layers 2
    "512_relu": {"hidden_sizes": [512]},
    "350_ReLU_350_ReLU": {"hidden_sizes": [350, 350]},
    "400_ReLU_128_ReLU": {"hidden_sizes": [400, 128]},
    # Encoder Type Models
    "5_ReLU": {"hidden_sizes": [5]},
    "512_ReLU-16_ReLU": {"hidden_sizes": [512, 256, 128, 64, 32, 16]},
    "256_ReLU_64_ReLU_128_ReLU": {"hidden_sizes": [256, 64, 128]},
    "512_ReLU_256_ReLU_64_ReLU_256_ReLU": {"hidden_sizes": [512, 256, 64, 256, 10]},
}


def test_model(dm, name, model):

    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=EPOCHS,
        gpus=AVAIL_GPUS,
        logger=TensorBoardLogger("A2_final_runs/", name=name),
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


def main():

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

    # # Test Standard FC Models
    # for name, layers in STD_MODELS.items():
    #     model = DenseNet(**layers, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    #     res = test_model(fashion_mnist_dm, name, model)
    #     results.append(res)

    # # Test the Conv Net
    # conv_net = ConvNet(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # res = test_model(fashion_mnist_dm, "conv_baseline", conv_net)
    # results.append(res)

    # # Models with KL Divergence
    # for name, layers in STD_MODELS.items():
    #     model = DenseNet(**layers, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, loss_fn='kl_div')
    #     res = test_model(fashion_mnist_dm, f"KL_{name}", model)
    #     results.append(res)


    # conv_net = ConvNet(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, loss_fn='kl_div')
    # res = test_model(fashion_mnist_dm, "KL_conv_baseline", conv_net)
    # results.append(res)

    # results = pd.DataFrame(results)
    # results.to_csv("results.csv")

    res_net = ResNet(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    res = test_model(fashion_mnist_dm, "resnet_50", res_net)
    results.append(res)

    results = pd.DataFrame(results)
    results.to_csv("resnet_results.csv")


if __name__ == "__main__":
    main()
