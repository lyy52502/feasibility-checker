"""
Conditional Variational Autoencoder (CVAE) for roll configuration.

  Condition c = mill requirements (what engineer specifies)
  Target    x = roll configuration (what model generates)

  During TRAINING:  encoder sees both c and x → learns latent z
  During INFERENCE: decoder takes c + sampled z → generates x
"""

import torch
import torch.nn as nn


class CVAE(nn.Module):
    def __init__(self, condition_dim: int, target_dim: int, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.target_dim = target_dim

        # ── Encoder: q(z | x, c) ─────────────────────────────────
        # Input: concatenation of target x and condition c
        enc_input = target_dim + condition_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_input, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        # Two heads: mean and log-variance of the latent distribution
        self.fc_mu     = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # ── Decoder: p(x | z, c) ─────────────────────────────────
        # Input: concatenation of latent z and condition c
        dec_input = latent_dim + condition_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_input, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, target_dim),
        )

        # ── Feasibility classifier head ───────────────────────────
        # Trained jointly: takes latent z → P(feasible)
        self.feasibility_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        h = self.encoder(torch.cat([x, c], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Reparameterisation trick: z = mu + eps * std"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor):
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        feasibility = self.feasibility_head(z)
        return x_recon, mu, logvar, feasibility

    def generate(self, c: torch.Tensor, n_samples: int = 5):
        """
        At inference time: given mill conditions c,
        generate n_samples candidate roll configurations.
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim)
            c_rep = c.unsqueeze(0).repeat(n_samples, 1)
            configs = self.decode(z, c_rep)
            feasibility = self.feasibility_head(z)
        return configs, feasibility


def cvae_loss(x_recon, x, mu, logvar, feasibility,
              beta: float = 1.0):
    """
    Total loss = Reconstruction + KL divergence + Feasibility

    Reconstruction: how well does the decoder recreate the config?
    KL:             how close is the latent distribution to N(0,1)?
    Feasibility:    all training samples are feasible (label = 1)
    """
    # MSE reconstruction loss
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')

    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Feasibility loss: all training data is feasible → label = 1
    ones = torch.ones_like(feasibility)
    feasibility_loss = nn.functional.binary_cross_entropy(
        feasibility, ones, reduction='mean'
    )

    total = recon_loss + beta * kl_loss + 0.1 * feasibility_loss
    return total, recon_loss, kl_loss, feasibility_loss