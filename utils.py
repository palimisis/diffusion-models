import os
from custom_datasets.sixray import SixRayDataSet

import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import CIFAR10
from torchvision.datasets.mnist import MNIST, FashionMNIST


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat(
            [
                torch.cat([i for i in images.cpu()], dim=-1),
            ],
            dim=-2,
        )
        .permute(1, 2, 0)
        .cpu()
    )
    plt.show()


def save_images(images, path, **kwargs):
    print("save images")
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.image_size, args.image_size)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(
                lambda t: (t * 2) - 1
            ),  # normalize to [-1, 1]
        ]
    )

    if args.dataset == "mnist":
        training_dataset = MNIST(
            root="./datasets", train=True, download=True, transform=transforms
        )
        validation_dataset = MNIST(
            root="./datasets", train=False, download=True, transform=transforms
        )
        n_classes = 10
    elif args.dataset == "fmnist":
        training_dataset = FashionMNIST(
            root="./datasets", train=True, download=True, transform=transforms
        )
        validation_dataset = FashionMNIST(
            root="./datasets", train=False, download=True, transform=transforms
        )
        n_classes = 10
    elif args.dataset == "cifar10":
        training_dataset = CIFAR10(
            root="./datasets", train=True, download=True, transform=transforms
        )
        validation_dataset = CIFAR10(
            root="./datasets", train=False, download=True, transform=transforms
        )
        n_classes = 10
    elif args.dataset == "100sports":
        training_dataset = datasets.ImageFolder(
            root="./datasets/100Sports/train", transform=transforms
        )
        validation_dataset = datasets.ImageFolder(
            root="./datasets/100Sports/valid", transform=transforms
        )
        n_classes = 100
    elif args.dataset == "sixray":
        training_dataset = SixRayDataSet(            
            root=args.dataset_path, transform=transforms
        )
        validation_dataset = SixRayDataSet(            
            root=args.dataset_path, transform=transforms
        )
        n_classes = 6
        return training_dataset, validation_dataset, n_classes

    print(len(training_dataset))

    training_dataloader = DataLoader(
        training_dataset, batch_size=args.batch_size, shuffle=True
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=True
    )
    return training_dataloader, validation_dataloader, n_classes


def local_setup(run_name):
    os.makedirs(f"models/{run_name}", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
