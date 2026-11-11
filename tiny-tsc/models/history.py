"""Ordered window embeddings and variable-length Perceiver histories."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class MicroDietStay(nn.Module):
    """Shared 24h embeddings and an ordered GRU; one prediction per causal prefix."""

    def __init__(self, seed):
        super().__init__()
        from models.mwdn_micro_diet import mWDN_micro_diet
        from models import seeded_model

        self.window = seeded_model(mWDN_micro_diet, 33, 128, seed)
        linear = self.window._model.head[0][1][0]
        assert isinstance(linear, nn.Linear) and linear.in_features == 48
        self.window._model.head[0][1][0] = nn.Identity()
        self.ordered = nn.GRU(49, 64, batch_first=True)
        self.classifier = nn.Linear(64, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, times, lengths):
        embeddings = self.window(x)
        sequences = torch.split(embeddings, lengths.tolist())
        padded = pad_sequence(sequences, batch_first=True)
        values = torch.cat((padded, times.unsqueeze(-1)), dim=-1)
        packed = pack_padded_sequence(
            values, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.ordered(packed)
        return self.classifier(hidden[-1])


class PerceiverStay(nn.Module):
    def __init__(self, config):
        super().__init__()
        from models.perceiver import PerceiverClassifier, PerceiverClassifierConfig

        self.classifier = PerceiverClassifier(PerceiverClassifierConfig(**config))
        self.loss_fn = self.classifier.loss_fn

    def forward(self, x, times, lengths):
        return self.classifier(x, times)
