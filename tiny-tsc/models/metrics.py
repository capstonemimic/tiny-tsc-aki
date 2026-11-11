import torch
import torch.nn as nn
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)


class ClassificationMetrics(nn.Module):
    def __init__(self, num_classes: int, prefix: str) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.prefix = prefix
        self.binary = num_classes == 2

        if self.binary:
            self.auroc = BinaryAUROC()
            self.auprc = BinaryAveragePrecision()
            self.f1 = BinaryF1Score()
            self.precision = BinaryPrecision()
            self.recall = BinaryRecall()
            self.specificity = BinarySpecificity()
            self.mcc = BinaryMatthewsCorrCoef()
        else:
            self.auroc = MulticlassAUROC(num_classes, average="macro")
            self.auprc = MulticlassAveragePrecision(num_classes, average="macro")
            self.f1 = MulticlassF1Score(num_classes, average="macro")
            self.precision = MulticlassPrecision(num_classes, average="macro")
            self.recall = MulticlassRecall(num_classes, average="macro")
            self.specificity = MulticlassSpecificity(num_classes, average="macro")
            self.mcc = MulticlassMatthewsCorrCoef(num_classes)

        self.confmat = MulticlassConfusionMatrix(num_classes)

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        preds = logits.argmax(dim=-1)
        if self.binary:
            prob_pos = torch.softmax(logits.float(), dim=-1)[..., 1]
            self.auroc.update(prob_pos, labels)
            self.auprc.update(prob_pos, labels)
        else:
            probs = torch.softmax(logits.float(), dim=-1)
            self.auroc.update(probs, labels)
            self.auprc.update(probs, labels)
        for metric in (
            self.f1,
            self.precision,
            self.recall,
            self.specificity,
            self.mcc,
        ):
            metric.update(preds, labels)
        self.confmat.update(preds, labels)

    def compute(self) -> dict[str, torch.Tensor]:
        cm = self.confmat.compute().float()
        per_class_acc = cm.diagonal() / cm.sum(dim=1).clamp_min(1.0)
        balanced_acc = per_class_acc.mean()
        overall_acc = cm.diagonal().sum() / cm.sum().clamp_min(1.0)

        metrics: dict[str, torch.Tensor] = {
            f"{self.prefix}_auroc": self.auroc.compute().detach(),
            f"{self.prefix}_auprc": self.auprc.compute().detach(),
            f"{self.prefix}_balanced_acc": balanced_acc,
            f"{self.prefix}_acc": overall_acc,
            f"{self.prefix}_f1": self.f1.compute().detach(),
            f"{self.prefix}_precision": self.precision.compute().detach(),
            f"{self.prefix}_recall": self.recall.compute().detach(),
            f"{self.prefix}_specificity": self.specificity.compute().detach(),
            f"{self.prefix}_mcc": self.mcc.compute().detach(),
        }
        for i in range(self.num_classes):
            metrics[f"{self.prefix}_acc_class_{i}"] = per_class_acc[i]
        return metrics

    def reset(self) -> None:
        for metric in (
            self.auroc,
            self.auprc,
            self.f1,
            self.precision,
            self.recall,
            self.specificity,
            self.mcc,
            self.confmat,
        ):
            metric.reset()
