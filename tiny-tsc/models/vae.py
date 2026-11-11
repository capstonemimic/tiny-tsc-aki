import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from models.base import BaseClassifier
from models.metrics import ClassificationMetrics
from models.mta import MultiTimeAttention, MultiTimeAttentionConfig, TemporalEmbedding


@dataclass(frozen=True, slots=True)
class MtanVaeClassifierConfig:
    d_input: int
    d_out: int = 2
    d_latent: int = 128
    n_hidden: int = 256
    n_hidden_dec: int = 50
    embed_time: int = 128
    n_heads: int = 1
    learn_embeddings: bool = True
    k_iwae: int = 1
    kl_anneal: bool = True
    kl_anneal_wait: int = 10
    kl_anneal_rate: float = 0.99
    sample_tp: float = 1.0
    n_division: int = 128
    alpha: float = 5.0
    norm: bool = True
    noise_std: float = 0.01
    noise_var: float | None = None


class MtanRnnEncoder(nn.Module):
    def __init__(
        self,
        d_input: int,
        query: torch.Tensor,
        d_latent: int = 2,
        n_hidden: int = 16,
        embed_time: int = 16,
        n_heads: int = 1,
        learn_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.dim = d_input
        self.n_hidden = n_hidden
        self.temporal_embedding = TemporalEmbedding(embed_time, learn_embeddings)
        self.register_buffer("query", query)
        self.att = MultiTimeAttention(
            MultiTimeAttentionConfig(2 * d_input, n_hidden, embed_time, n_heads)
        )
        self.gru_rnn = nn.GRU(n_hidden, n_hidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2 * n_hidden, 50), nn.ReLU(), nn.Linear(50, d_latent * 2)
        )

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        mask = x[:, :, self.dim :]
        mask = torch.cat((mask, mask), 2)
        key = self.temporal_embedding(time_steps)
        query = self.temporal_embedding(self.query.unsqueeze(0))
        out = self.att(query, key, x, mask)
        out, _ = self.gru_rnn(out)
        return self.hiddens_to_z0(out)


class MtanRnnDecoder(nn.Module):
    def __init__(
        self,
        d_input: int,
        query: torch.Tensor,
        d_latent: int = 2,
        n_hidden: int = 16,
        embed_time: int = 16,
        n_heads: int = 1,
        learn_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.dim = d_input
        self.n_hidden = n_hidden
        self.temporal_embedding = TemporalEmbedding(embed_time, learn_embeddings)
        self.register_buffer("ref_query", query)
        self.att = MultiTimeAttention(
            MultiTimeAttentionConfig(2 * n_hidden, 2 * n_hidden, embed_time, n_heads)
        )
        self.gru_rnn = nn.GRU(d_latent, n_hidden, bidirectional=True, batch_first=True)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2 * n_hidden, 50), nn.ReLU(), nn.Linear(50, d_input)
        )

    def forward(self, z: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru_rnn(z)
        query = self.temporal_embedding(time_steps)
        key = self.temporal_embedding(self.ref_query.unsqueeze(0))
        out = self.att(query, key, out)
        return self.z0_to_obs(out)


class LatentClassifier(nn.Module):
    def __init__(self, d_latent: int, n_hidden: int = 16, d_out: int = 2) -> None:
        super().__init__()
        self.gru_rnn = nn.GRU(d_latent, n_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(n_hidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, d_out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _, out = self.gru_rnn(z)
        return self.classifier(out.squeeze(0))


def subsample_timepoints(
    data: torch.Tensor,
    time_steps: torch.Tensor,
    mask: torch.Tensor,
    sample_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = data.shape[1]
    n_samples = max(1, int(seq_len * sample_ratio))
    indices = torch.randperm(seq_len)[:n_samples].sort()[0]
    return data[:, indices, :], time_steps[:, indices], mask[:, indices, :]


def gaussian_log_likelihood(
    mu: torch.Tensor, data: torch.Tensor, mask: torch.Tensor, std: float = 0.01
) -> torch.Tensor:
    if std <= 0:
        raise ValueError(f"noise std must be positive, got {std}")
    mu_f = mu.float()
    data_f = data.float().unsqueeze(0).expand_as(mu_f)
    mask_f = mask.float().unsqueeze(0).expand_as(mu_f)
    noise_logvar = 2.0 * math.log(std)
    log_pdf = -0.5 * (
        math.log(2.0 * math.pi)
        + noise_logvar
        + (data_f - mu_f) ** 2 / math.exp(noise_logvar)
    )
    return torch.sum(log_pdf * mask_f, dim=(2, 3))


def kl_divergence_normal(
    qz0_mean: torch.Tensor, qz0_logvar: torch.Tensor
) -> torch.Tensor:
    mean_f = qz0_mean.float()
    logvar_f = qz0_logvar.float()
    q = Normal(mean_f, torch.exp(0.5 * logvar_f))
    p = Normal(torch.zeros_like(mean_f), torch.ones_like(logvar_f))
    return torch.sum(kl_divergence(q, p), dim=(1, 2))


def compute_losses(
    observed_data: torch.Tensor,
    observed_mask: torch.Tensor,
    qz0_mean: torch.Tensor,
    qz0_logvar: torch.Tensor,
    pred_x: torch.Tensor,
    norm: bool = False,
    noise_std: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    logpx = gaussian_log_likelihood(pred_x, observed_data, observed_mask, std=noise_std)
    analytic_kl = kl_divergence_normal(qz0_mean, qz0_logvar).unsqueeze(0)
    if norm:
        n_observed = observed_mask.float().sum(dim=(1, 2)).clamp_min(1.0)
        logpx = logpx / n_observed.unsqueeze(0)
        analytic_kl = analytic_kl / n_observed.unsqueeze(0)
    return logpx, analytic_kl


def masked_mse(
    data: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mse = F.mse_loss(pred, data, reduction="none")
    return (mse * mask).sum() / (mask.sum() + 1e-8)


class MtanVaeClassifier(BaseClassifier):
    def __init__(
        self, config: MtanVaeClassifierConfig, learning_rate: float | None = 1e-4
    ) -> None:
        super().__init__(config, learning_rate)
        self.ref_query: torch.Tensor
        self.register_buffer("ref_query", torch.linspace(0, 1, config.n_division))

        self.encoder = MtanRnnEncoder(
            d_input=config.d_input,
            query=self.ref_query,
            d_latent=config.d_latent,
            n_hidden=config.n_hidden,
            embed_time=config.embed_time,
            n_heads=config.n_heads,
            learn_embeddings=config.learn_embeddings,
        )
        self.decoder = MtanRnnDecoder(
            d_input=config.d_input,
            query=self.ref_query,
            d_latent=config.d_latent,
            n_hidden=config.n_hidden_dec,
            embed_time=config.embed_time,
            n_heads=config.n_heads,
            learn_embeddings=config.learn_embeddings,
        )
        self.classifier = LatentClassifier(
            d_latent=config.d_latent, n_hidden=config.n_hidden, d_out=config.d_out
        )
        self.ce_criterion = nn.CrossEntropyLoss()

    def get_kl_coef(self) -> float:
        if not self.config.kl_anneal:
            return 1.0
        epoch = self.current_epoch + 1
        if epoch < self.config.kl_anneal_wait:
            return 0.0
        return 1.0 - self.config.kl_anneal_rate ** (epoch - self.config.kl_anneal_wait)

    def _noise_std(self) -> float:
        if self.config.noise_var is not None:
            return math.sqrt(self.config.noise_var)
        return self.config.noise_std

    def encode(
        self, data: torch.Tensor, time_steps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(data, time_steps)
        d_latent = self.config.d_latent
        qz0_mean = out[:, :, :d_latent]
        qz0_logvar = out[:, :, d_latent:].clamp(-15.0, 15.0)
        return qz0_mean, qz0_logvar

    def reparameterize(
        self, mean: torch.Tensor, logvar: torch.Tensor, k_iwae: int = 1
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(k_iwae, *mean.shape, device=mean.device)
        return (mean + std * eps).view(-1, mean.shape[1], mean.shape[2])

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        qz0_mean, qz0_logvar = self.encode(x, time_steps)
        z0 = (
            self.reparameterize(qz0_mean, qz0_logvar, k_iwae=1)
            if self.training
            else qz0_mean
        )
        return self.classifier(z0)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, time_steps, labels = batch
        dim = self.config.d_input
        batch_len = x.shape[0]
        k_iwae = self.config.k_iwae

        observed_data = x[:, :, :dim]
        observed_mask = x[:, :, dim : 2 * dim]

        if self.config.sample_tp < 1.0:
            sub_data, sub_tp, sub_mask = subsample_timepoints(
                observed_data.clone(),
                time_steps.clone(),
                observed_mask.clone(),
                self.config.sample_tp,
            )
        else:
            sub_data, sub_tp, sub_mask = observed_data, time_steps, observed_mask

        encoder_input = torch.cat((sub_data, sub_mask), 2)
        qz0_mean, qz0_logvar = self.encode(encoder_input, sub_tp)
        z0 = self.reparameterize(qz0_mean, qz0_logvar, k_iwae=k_iwae)

        pred_y = self.classifier(z0)
        time_steps_rep = (
            time_steps[None].repeat(k_iwae, 1, 1).view(-1, time_steps.shape[1])
        )
        pred_x = self.decoder(z0, time_steps_rep).view(k_iwae, batch_len, -1, dim)

        logpx, analytic_kl = compute_losses(
            observed_data,
            observed_mask,
            qz0_mean,
            qz0_logvar,
            pred_x,
            norm=self.config.norm,
            noise_std=self._noise_std(),
        )
        kl_coef = self.get_kl_coef()
        recon_loss = -(
            torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean()
            - math.log(k_iwae)
        )

        labels_rep = labels.unsqueeze(0).repeat_interleave(k_iwae, 0).view(-1)
        ce_loss = self.ce_criterion(pred_y, labels_rep)
        loss = recon_loss + self.config.alpha * ce_loss

        self.train_metrics.update(pred_y[:batch_len].detach(), labels.detach())
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_len,
        )
        self.log(
            "train_recon_loss",
            recon_loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_len,
        )
        self.log(
            "train_ce_loss", ce_loss, on_step=True, on_epoch=True, batch_size=batch_len
        )
        self.log(
            "train_mse",
            masked_mse(observed_data, pred_x.mean(0), observed_mask),
            on_step=True,
            on_epoch=True,
            batch_size=batch_len,
        )
        self.log(
            "kl_coef",
            kl_coef,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_len,
        )

        return loss

    def _eval_step(
        self, batch, metrics: ClassificationMetrics, prefix: str
    ) -> torch.Tensor:
        x, time_steps, labels = batch
        dim = self.config.d_input
        batch_len = x.shape[0]

        observed_data = x[:, :, :dim]
        observed_mask = x[:, :, dim : 2 * dim]
        encoder_input = torch.cat((observed_data, observed_mask), 2)

        qz0_mean, qz0_logvar = self.encode(encoder_input, time_steps)
        # Validation and test metrics must be repeatable. Use the posterior
        # mean rather than drawing a new latent sample on every evaluation.
        z0 = qz0_mean
        logits = self.classifier(z0)

        ce_loss = self.ce_criterion(logits, labels)
        pred_x = self.decoder(z0, time_steps).unsqueeze(0)
        logpx, analytic_kl = compute_losses(
            observed_data,
            observed_mask,
            qz0_mean,
            qz0_logvar,
            pred_x,
            norm=self.config.norm,
            noise_std=self._noise_std(),
        )
        recon_loss = -(logpx - analytic_kl).mean()
        loss = recon_loss + self.config.alpha * ce_loss

        metrics.update(logits, labels)
        self.log(
            f"{prefix}_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_len
        )
        self.log(
            f"{prefix}_recon_loss", recon_loss, on_epoch=True, batch_size=batch_len
        )
        self.log(f"{prefix}_ce_loss", ce_loss, on_epoch=True, batch_size=batch_len)
        self.log(
            f"{prefix}_mse",
            masked_mse(observed_data, pred_x.squeeze(0), observed_mask),
            on_epoch=True,
            batch_size=batch_len,
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, self.val_metrics, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, self.test_metrics, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
