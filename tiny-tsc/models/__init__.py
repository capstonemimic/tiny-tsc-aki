"""Registry of the 19 window models and the Micro-Diet history classifier."""

import types
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

REGULAR = (
    "mwdn",
    "tsitplus",
    "gmlp",
    "gmlp_moe",
    "minirocket",
    "tsperceiver",
    "tcn",
    "tstplus",
    "inceptiontime",
    "omniscale",
    "rescnn",
    "mwdn_diet",
    "mwdn_micro",
    "mwdn_micro_diet",
)
IRREGULAR = ("perceiver", "mtan_vae", "mamba", "neural_cde", "ode_rnn")
WINDOW_MODELS = (*REGULAR[:11], *IRREGULAR, *REGULAR[11:])
ALL_MODELS = (*WINDOW_MODELS, "micro_diet_gru")
HISTORY_MODELS = ("perceiver", "micro_diet_gru")


def seeded_model(factory, channels, length, seed):
    """Use the recorded wavelet initializer with an explicit local RNG."""
    from models import _wavelet

    rng = np.random.default_rng(seed)
    original = _wavelet.create_banded_weight

    def weight(seq_len, filter_values):
        result = (
            rng.standard_normal((seq_len, seq_len))
            * 0.1
            * np.min(np.abs(filter_values))
        )
        for i in range(seq_len):
            for j_idx, j in enumerate(range(i, min(i + len(filter_values), seq_len))):
                result[i, j] = filter_values[j_idx]
        return result

    try:
        _wavelet.create_banded_weight = weight
        return factory(channels, 2, length)
    finally:
        _wavelet.create_banded_weight = original


def multilevel_forward(self, x):
    outputs = []
    for block in self.blocks:
        x, features = block(x)
        outputs.append(features)
    return self._model(torch.cat(outputs, dim=-1))


def make_model(name, seed=4242, compiled=False, model_config=None):
    if name not in ALL_MODELS:
        raise ValueError(f"Unknown model: {name}")
    if name in REGULAR:
        import tsai.all as tsai
        from models.gmlp_moe import gMLP_MoE
        from models.mwdn_diet import mWDN_diet
        from models.mwdn_micro import mWDN_micro
        from models.mwdn_micro_diet import mWDN_micro_diet

        factories = {
            "mwdn": tsai.mWDN,
            "tsitplus": tsai.TSiTPlus,
            "gmlp": tsai.gMLP,
            "gmlp_moe": gMLP_MoE,
            "minirocket": tsai.MiniRocket,
            "tsperceiver": tsai.TSPerceiver,
            "tstplus": tsai.TSTPlus,
            "omniscale": tsai.OmniScaleCNN,
            "mwdn_diet": mWDN_diet,
            "mwdn_micro": mWDN_micro,
            "mwdn_micro_diet": mWDN_micro_diet,
            "tcn": lambda c, o, t: tsai.TCN(c, o),
            "rescnn": lambda c, o, t: tsai.ResCNN(c, o),
            "inceptiontime": lambda c, o, t: tsai.InceptionTime(c, o),
        }
        model = seeded_model(factories[name], 33, 128, seed)
        if name == "mwdn":
            model.forward = types.MethodType(multilevel_forward, model)
        return model
    if name == "micro_diet_gru":
        from models.history import MicroDietStay

        return MicroDietStay(seed)
    if name == "ode_rnn":
        from models.ode_rnn import ODERNN

        model = ODERNN(33)
    else:
        from models.perceiver import PerceiverClassifierConfig
        from models.history import PerceiverStay
        from models.mamba import CTMambaClassifier, CTMambaClassifierConfig
        from models.neural_cde import NeuralCDEClassifier, NeuralCDEClassifierConfig
        from models.vae import MtanVaeClassifier, MtanVaeClassifierConfig

        configs = yaml.safe_load(
            (Path(__file__).parents[1] / "configs/models.yaml").read_text()
        )
        config = model_config or configs[name]
        if name == "perceiver":
            PerceiverClassifierConfig(**config)
            return PerceiverStay(config)
        cls, cfg = {
            "mamba": (CTMambaClassifier, CTMambaClassifierConfig),
            "neural_cde": (NeuralCDEClassifier, NeuralCDEClassifierConfig),
            "mtan_vae": (MtanVaeClassifier, MtanVaeClassifierConfig),
        }[name]
        model = cls(cfg(**config))
    from models.recurrences import accelerate

    return accelerate(model, name, compiled=compiled)


def forward(model, name, batch):
    if name in REGULAR:
        return model(batch["x"])
    if name in HISTORY_MODELS:
        return model(batch["x"], batch["times"], batch["lengths"])
    return model(batch["x"], batch["times"])


def loss(model, name, batch):
    if name != "mtan_vae":
        logits = forward(model, name, batch)
        return (
            F.cross_entropy(logits, batch["labels"])
            if name in REGULAR or name == "ode_rnn"
            else model.loss_fn(logits, batch["labels"])
        )
    from models.vae import compute_losses

    x, times, y = batch["x"], batch["times"], batch["labels"]
    mean, logvar = model.encode(x, times)
    z = model.reparameterize(mean, logvar, 1)
    logits = model.classifier(z)
    prediction = model.decoder(z, times).unsqueeze(0)
    c = model.config.d_input
    logp, kl = compute_losses(
        x[:, :, :c],
        x[:, :, c:],
        mean,
        logvar,
        prediction,
        norm=model.config.norm,
        noise_std=model.config.noise_std,
    )
    return -logp.mean() + kl.mean() + model.config.alpha * F.cross_entropy(logits, y)
