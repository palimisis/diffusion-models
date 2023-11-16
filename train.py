import logging
import os

import torch
from torch import nn, optim
from tqdm import tqdm

import wandb
from diffusion import Diffusion
from eval import eval
from unet import UNet
from utils import get_data, local_setup, save_images

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_step(model, diffusion, optimizer, loss_fn, images, labels):
    images = images.to(device)
    labels = labels.to(device)
    t = diffusion.sample_timesteps(images.shape[0]).to(device)
    x_t, noise = diffusion.add_noise(images, t)
    predicted_noise = model(x_t, t, x_self_cond=labels)
    loss = loss_fn(noise, predicted_noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def valid_step(model, diffusion, loss_fn, images, labels):
    images = images.to(device)
    labels = labels.to(device)
    t = diffusion.sample_timesteps(images.shape[0]).to(device)
    x_t, noise = diffusion.add_noise(images, t)
    predicted_noise = model(x_t, t, x_self_cond=labels)
    loss = loss_fn(noise, predicted_noise)
    return loss


def train(args):
    local_setup(args.run_name)
    wandb.init(project="diffusion_models", config=args)

    training_dataloader, validation_dataloader, n_classes = get_data(args)
    # model = UNet(c_in=args.channels, c_out=args.channels, device=device).to(device)
    # model = UNet().to(device)
    print(args.class_condition)
    model = UNet(
        dim=args.image_size,
        channels=args.channels,
        self_condition=args.class_condition,
        num_classes=n_classes,
    ).to(device)

    # print total model parameters
    logging.info(
        f"Total number of parameters: {sum(p.numel() for p in model.parameters())}"
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(
        img_size=args.image_size,
        device=device,
        channels=args.channels,
        n_classes=n_classes,
    )
    l = len(training_dataloader)
    # l = 1

    for epoch in range(args.epochs):
        # pbar = tqdm(training_dataloader, total=1)
        pbar = tqdm(training_dataloader)
        # Training 1 epoch
        model.train()
        for i, (images, labels) in enumerate(pbar):
            loss = train_step(model, diffusion, optimizer, mse, images, labels)
            pbar.set_postfix(MSE=loss.item())
            # if i == l - 1:
            #     break

        wandb.log({"loss": loss.item()}, step=epoch + 1)

        pbar = tqdm(validation_dataloader, total=1)
        # Validation
        model.eval()
        for i, (images, labels) in enumerate(pbar):
            vall_loss = valid_step(model, diffusion, mse, images, labels)
            pbar.set_postfix(MSE=vall_loss.item())
            if i == l - 1:
                break
        wandb.log({"val_loss": vall_loss.item()}, step=epoch + 1)

        # Sampling
        sampled_images = diffusion.sample(model, samples_per_class=1)
        save_images(
            sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg")
        )

        # Log images to wandb
        if (epoch + 1) % 5 == 0:
            wandb.log(
                {f"epoch_{epoch+1}_samples": [wandb.Image(i) for i in sampled_images]},
                step=epoch + 1,
            )

        torch.save(
            model.state_dict(),
            os.path.join(f"models", args.run_name, f"ckpt_epoch{epoch}.pt"),
        )

    args.cpt_path = os.path.join(f"models", args.run_name, f"ckpt_epoch{epoch}.pt")
    eval(args, training_dataset=training_dataloader, n_classes=n_classes)

    wandb.finish()


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="DDPM")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument(
        "--class_condition", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    args.lr = 1e-4

    train(args)


if __name__ == "__main__":
    launch()
