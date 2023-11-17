import logging

import numpy as np
import torch
from tqdm import tqdm

import noise_schedulers

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=64,
        device="cuda",
        schedule="linear",
        channels=3,
        n_classes=10,
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.schedule = schedule
        self.channels = channels
        self.n_classes = n_classes

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        if self.schedule == "linear":
            return noise_schedulers.linear_beta_schedule(
                self.noise_steps, self.beta_start, self.beta_end
            )
        elif self.schedule == "cosine":
            return noise_schedulers.cosine_beta_schedule(self.noise_steps, s=0.008)
        elif self.schedule == "quadratic":
            return noise_schedulers.quadratic_beta_scedule(
                self.noise_steps, self.beta_start, self.beta_end
            )
        elif self.schedule == "sigmoid":
            return noise_schedulers.sigmoid_beta_schedule(
                self.noise_steps, self.beta_start, self.beta_end
            )

    def add_noise(self, x, t):
        """
        Add noise to the input image x at time t in a single timestep.
        Eq (4) in DDPM paper.
        """
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar[t])[
            :, None, None, None
        ]
        e = torch.randn_like(x)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * e, e

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, samples_per_class=8):
        n = self.n_classes * samples_per_class
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.channels, self.img_size, self.img_size)).to(
                self.device
            )
            y = (
                torch.tensor([[i] * samples_per_class for i in range(self.n_classes)])
                .flatten()
                .to(self.device)
            )
            for i in tqdm(reversed(range(1, self.noise_steps))):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, x_self_cond=y)

                alpha = self.alpha[t][:, None, None, None]
                alpha_bar = self.alpha_bar[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
