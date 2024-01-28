import logging
import os
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

import wandb
from diffusion import Diffusion
from eval import eval
from unet import UNet
from utils import get_data, local_setup, save_images
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print(available_gpus)

def save_model_checkpoint(model, epoch, optimizer, path: Path, loss, step):
    global nth_batch
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "nth_batch": nth_batch,
            "step": step
        },
        path / "checkpoint.chk",
    )


def train_step(model, diffusion, optimizer, loss_fn, images, labels=None):
    images = images.to(device)
    if labels: labels = labels.to(device)
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

nth_batch = 0
def get_batch(data, batch_size=4, image_size=768):
    global nth_batch
    nth_batch += 1

    total_batches = len(data) // batch_size
    if nth_batch >= total_batches:
        nth_batch = 0

    batch = np.zeros((batch_size, 3, image_size, image_size))
    for i in range(batch_size):
        batch[i] = data[i*nth_batch]
    return torch.as_tensor(batch, dtype=torch.float32)


def train(args):
    local_setup(args.run_name)
    wandb.init(project="sixray_ddpm_from_scratch", config=args)

    # training_dataloader, validation_dataloader, n_classes = get_data(args)
    training_dataset, validation_dataset, n_classes = get_data(args)

    model = UNet(
        dim=args.image_size,
        channels=args.channels,
        self_condition=args.class_condition,
        num_classes=10,
    ).to(device)


    # print total model parameters
    logging.info(
        f"Total number of parameters: {sum(p.numel() for p in model.parameters())}"
    )
    print(f"Total batches: {len(training_dataset//args.batch_size)}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(
        img_size=args.image_size,
        device=device,
        channels=args.channels,
        n_classes=None,
    )

    chk_path = Path(args.cpt_path)
    if chk_path is not None:
        if os.path.exists(chk_path / "checkpoint.chk"):
            print(f"Loaded checkpoint from {chk_path/'checkpoint.chk'}")
            checkpoint = torch.load(chk_path / "checkpoint.chk")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            current_epoch = checkpoint["epoch"]
            loss_item = checkpoint["loss"]
            nth_batch = checkpoint["nth_batch"]
            step = checkpoint["step"]
        else:
            if not os.path.exists(chk_path):
                os.mkdir(chk_path)
            current_epoch = 1
            step = 1
    else:
        current_epoch = 1
        step=1
    
    print(f"Starting step: {step}")
    print(f"Nth batch: {nth_batch if nth_batch else 'nth batch is null'}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        for i in range(len(training_dataset) // args.batch_size):
            images = get_batch(training_dataset, batch_size=args.batch_size, image_size=args.image_size)
            # print(f"Images shape: {images.shape}")
            loss = train_step(model, diffusion, optimizer, mse, images)

            print(f"Step: {step} -- Loss: {loss.item()}")

            wandb.log({"loss": loss.item()}, step=step)
            step+=1

            # Log images to wandb
            if step % 2000 == 0:
                print("Sampling new images...")
                # Sampling
                sampled_images = diffusion.sample(model, samples_per_class=2)
                save_images(
                    sampled_images, os.path.join("results", args.run_name, f"{step}.jpg")
                )
                wandb.log(
                    {f"step_{step}_samples": [wandb.Image(i) for i in sampled_images]},
                    step=step,
                )

                save_model_checkpoint(
                    model,
                    current_epoch,
                    optimizer,
                    chk_path,
                    loss,
                    step
                )
    wandb.finish()


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="DDPM")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument(
        "--class_condition", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--cpt_path",
        action="store",
        type=str,
        default="checkpoints",
        help="Path for saving training checkpoints",
    )
    args = parser.parse_args()
    args.lr = 1e-4

    train(args)


if __name__ == "__main__":
    launch()
