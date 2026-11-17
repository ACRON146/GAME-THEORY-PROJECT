from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def privacy_gap_loss(member_scores: torch.Tensor, reference_scores: torch.Tensor) -> torch.Tensor:
    """Differentiable squared mean discriminator-score gap.

    The reference set is used by this training objective. It must not also be
    described as an untouched membership-inference non-member evaluation set.
    """
    if member_scores.numel() == 0 or reference_scores.numel() == 0:
        raise ValueError("Privacy regularization needs non-empty member and reference batches.")
    return (member_scores.mean() - reference_scores.mean()).square()


class TabularGAN:
    """Compact GAN with a discriminator member/reference score-gap penalty.

    ``lam=0`` is the ordinary GAN objective. For positive ``lam``, the
    discriminator objective is real/fake BCE plus ``lam * privacy_gap_loss``.
    This is an empirical regularizer, not a differential privacy mechanism.
    """

    def __init__(self, input_dim, latent_dim=32, hidden_dim=64, lr=2e-4, seed=11):
        if int(input_dim) <= 0 or int(latent_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("input_dim, latent_dim, and hidden_dim must be positive.")
        if not math.isfinite(float(lr)) or float(lr) <= 0:
            raise ValueError("learning rate must be finite and positive.")
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = torch.device("cpu")
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        self.g = Generator(self.latent_dim, int(hidden_dim), self.input_dim).to(self.device)
        self.d = Discriminator(self.input_dim, int(hidden_dim)).to(self.device)
        self.g_opt = torch.optim.Adam(self.g.parameters(), lr=float(lr), betas=(0.5, 0.9))
        self.d_opt = torch.optim.Adam(self.d.parameters(), lr=float(lr), betas=(0.5, 0.9))
        self.training_history: list[dict[str, float]] = []

    def fit(self, members, reference, lam=0.0, epochs=10, batch_size=128):
        lam = float(lam)
        epochs, batch_size = int(epochs), int(batch_size)
        if not math.isfinite(lam) or lam < 0:
            raise ValueError("lambda must be finite and non-negative.")
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive integers.")
        member = torch.as_tensor(np.asarray(members), dtype=torch.float32, device=self.device)
        reference = torch.as_tensor(np.asarray(reference), dtype=torch.float32, device=self.device)
        for name, values in (("members", member), ("reference", reference)):
            if values.ndim != 2 or values.shape[1] != self.input_dim:
                raise ValueError(f"{name} must be a 2D matrix with {self.input_dim} columns.")
            if len(values) < 2:
                raise ValueError("GAN requires at least two members and reference records.")
            if not torch.isfinite(values).all():
                raise ValueError(f"{name} contain non-finite values.")

        bce = nn.BCEWithLogitsLoss()
        start = time.perf_counter()
        n = len(member)
        history: list[dict[str, float]] = []
        self.g.train()
        self.d.train()
        for epoch in range(epochs):
            order = torch.randperm(n)
            totals = {"discriminator_adversarial_loss": 0.0, "privacy_gap_loss": 0.0,
                      "weighted_privacy_loss": 0.0, "generator_loss": 0.0}
            steps = 0
            for ids in order.split(batch_size):
                real = member[ids]
                ref_ids = torch.randint(0, len(reference), (len(ids),), device=self.device)
                ref_batch = reference[ref_ids]
                fake = self.g(torch.randn(len(ids), self.latent_dim, device=self.device)).detach()

                d_real = self.d(real)
                d_fake = self.d(fake)
                adversarial_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
                # Keep this computation connected to D parameters; do not detach scores.
                gap_loss = privacy_gap_loss(d_real, self.d(ref_batch))
                weighted_gap = lam * gap_loss
                d_loss = adversarial_loss + weighted_gap
                if not torch.isfinite(d_loss):
                    raise FloatingPointError(f"Non-finite discriminator loss at epoch {epoch + 1}.")
                self.d_opt.zero_grad(set_to_none=True)
                d_loss.backward()
                self.d_opt.step()

                generated = self.g(torch.randn(len(ids), self.latent_dim, device=self.device))
                g_loss = bce(self.d(generated), torch.ones(len(ids), 1, device=self.device))
                if not torch.isfinite(g_loss):
                    raise FloatingPointError(f"Non-finite generator loss at epoch {epoch + 1}.")
                self.g_opt.zero_grad(set_to_none=True)
                g_loss.backward()
                self.g_opt.step()

                totals["discriminator_adversarial_loss"] += float(adversarial_loss.detach())
                totals["privacy_gap_loss"] += float(gap_loss.detach())
                totals["weighted_privacy_loss"] += float(weighted_gap.detach())
                totals["generator_loss"] += float(g_loss.detach())
                steps += 1
            summary = {name: value / max(steps, 1) for name, value in totals.items()}
            summary.update({"epoch": float(epoch + 1), "lambda": lam, "steps": float(steps)})
            history.append(summary)
        self.training_history = history
        self.g.eval()
        self.d.eval()
        elapsed = time.perf_counter() - start
        if not math.isfinite(elapsed):
            raise FloatingPointError("Training runtime is non-finite.")
        return elapsed

    @torch.no_grad()
    def sample(self, n: int, batch_size=1024) -> np.ndarray:
        n, batch_size = int(n), int(batch_size)
        if n < 0 or batch_size <= 0:
            raise ValueError("sample count must be non-negative and batch_size positive.")
        self.g.eval()
        parts = []
        for start in range(0, n, batch_size):
            count = min(batch_size, n - start)
            parts.append(self.g(torch.randn(count, self.latent_dim, device=self.device)).cpu().numpy())
        return np.concatenate(parts, axis=0) if parts else np.empty((0, self.input_dim), dtype=np.float32)
