import logging
import os

import torch
from torch import nn, optim
from tqdm import tqdm

import wandb
from diffusion import Diffusion
from unet import UNet
from utils import get_data, local_setup, save_images


def train(args):
    local_setup(args.run_name)
    wandb.init(project="diffusion_models", name=args.run_name, config=args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader = get_data(args)
    # model = UNet(c_in=args.channels, c_out=args.channels, device=device).to(device)
    # model = UNet().to(device)
    model = UNet(dim=args.image_size, channels=args.channels).to(device)

    # print total model parameters
    logging.info(
        f"Total number of parameters: {sum(p.numel() for p in model.parameters())}"
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(
        img_size=args.image_size, device=device, channels=args.channels
    )
    l = len(dataloader)
    # l = 1

    for epoch in range(args.epochs):
        # pbar = tqdm(dataloader, total=1)
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.add_noise(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            # logging.info("MSE": loss.item())

            # if i == l - 1:
            #     break

        wandb.log({"MSE": loss.item()}, step=epoch + 1)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(
            sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg")
        )
        # Log images to wandb
        if (epoch + 1) % 5 == 0:
            wandb.log(
                {f"epoch_{epoch+1}_samples": [wandb.Image(i) for i in sampled_images]},
                step=epoch + 1,
            )

        torch.save(model.state_dict(), os.path.join("models", args.run_name, "ckpt.pt"))

    wandb.finish()


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="DDPM")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="mnist")
    args = parser.parse_args()
    args.lr = 1e-4

    train(args)


if __name__ == "__main__":
    launch()
