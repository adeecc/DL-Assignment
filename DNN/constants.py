import os

import torch

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 50
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 128
NUM_WORKERS = int(os.cpu_count() / 2)


classes = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
)
