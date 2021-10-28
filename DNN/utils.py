import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


from constants import classes


def matplotlib_imshow(img: torch.Tensor):
    img = img.squeeze(dim=0)
    npimg = img.cpu().numpy()
    plt.imshow(npimg, cmap="Greys")


def images_to_probs(net: nn.Module, images):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    output = net(images)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(8, 10))
    for idx in range(16):
        ax = fig.add_subplot(4, 4, idx + 1, xticks=[], yticks=[])

        matplotlib_imshow(images[idx])
        ax.set_title(
            f"{classes[preds[idx]]}, {probs[idx] * 100.0:.1f}%\n(label: {classes[labels[idx]]})",
            color=("green" if preds[idx] == labels[idx].item() else "red"),
        )
    return fig


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)