import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.metrics import ClassificationMetrics


class BaseClassifier(L.LightningModule):
    def __init__(self, config, learning_rate: float | None = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.learning_rate = learning_rate if learning_rate is not None else 1e-3

        self.num_classes: int = config.d_out
        self.train_metrics = ClassificationMetrics(self.num_classes, "train")
        self.val_metrics = ClassificationMetrics(self.num_classes, "val")
        self.test_metrics = ClassificationMetrics(self.num_classes, "test")

        self.loss_fn: nn.Module = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, time_steps, labels = batch
        logits = self(x, time_steps)
        loss = self.loss_fn(logits, labels)
        self.train_metrics.update(logits, labels)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.size(0),
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), prog_bar=False)
        self.train_metrics.reset()

    def _eval_step(
        self, batch, metrics: ClassificationMetrics, prefix: str
    ) -> torch.Tensor:
        x, time_steps, labels = batch
        logits = self(x, time_steps)
        loss = F.cross_entropy(logits, labels)
        metrics.update(logits, labels)
        self.log(
            f"{prefix}_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, self.val_metrics, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, self.test_metrics, "test")

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        auroc = metrics.pop("val_auroc", None)
        self.log_dict(metrics, prog_bar=False)
        if auroc is not None:
            self.log("val_auroc", auroc, prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), prog_bar=False)
        self.test_metrics.reset()

    def configure_optimizers(self):
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ("bias", "norm")):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        weight_decay = getattr(self.config, "weight_decay", 0.01)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.trainer.max_epochs),
            eta_min=self.learning_rate * 0.01,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
