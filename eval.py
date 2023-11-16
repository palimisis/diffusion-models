"""
Calculate FID score for a given diffusion model.
"""
import os
import subprocess

import torch
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance

from diffusion import Diffusion
from unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(diffusion: Diffusion, model, dataset, n_classes):
    # Generate samples
    samples_per_class = 10
    sample_images = diffusion.sample(model, samples_per_class=samples_per_class)

    # For each class in the dataset, calculate FID
    # TODO: Optimize this
    for i in range(n_classes):
        images = []
        for img, label in dataset:
            idxs = torch.where(label == i)[0]
            images.extend(img[idxs])

            if len(images) > samples_per_class:
                images = images[:samples_per_class]
                break

        real_images = torch.stack(images)

        fid = FrechetInceptionDistance(normalize=True)
        fid.update(real_images, real=True)
        fid.update(
            sample_images[
                i * samples_per_class : i * samples_per_class + samples_per_class
            ],
            real=False,
        )
        print(f"Class {i} FID: {float(fid.compute())}")


def eval(args, training_dataset, n_classes):
    model = UNet(
        dim=args.image_size,
        channels=args.channels,
        self_condition=args.class_condition,
        num_classes=n_classes,
    ).to(device)
    model.load_state_dict(torch.load(args.cpt_path))
    print(sum(p.numel() for p in model.parameters()))
    diffusion = Diffusion(
        img_size=args.image_size,
        device=device,
        channels=args.channels,
        n_classes=n_classes,
    )

    evaluate(diffusion, model, training_dataset, n_classes)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument(
        "--cpt_path", type=str, default="./models/ddpm_cifar10/ckpt_epoch2.pt"
    )
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--class_condition", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    eval(args)


if __name__ == "__main__":
    main()
