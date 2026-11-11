"""Equivalent parallel state-space and fixed-step CDE recurrences."""

import types

import torch
from torch import nn
from torch.nn import functional as F


def parallel_ssm(self, x, time_gaps, padding_mask=None):
    if padding_mask is not None:
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
    length = x.shape[1]
    xc = F.silu(self.conv1d(x.transpose(1, 2))[:, :, :length].transpose(1, 2))
    projected = self.x_proj(xc)
    r, n = self.dt_proj.in_features, self.d_state
    delta = F.softplus(
        self.dt_proj(projected[:, :, :r]) + self.time_proj(time_gaps.unsqueeze(-1))
    )
    a = torch.exp(delta.unsqueeze(-1) * -torch.exp(self.A_log)[None, None])
    b = (
        delta.unsqueeze(-1)
        * projected[:, :, r : r + n].unsqueeze(-2)
        * xc.unsqueeze(-1)
    )
    offset = 1
    # Inclusive scan of affine transformations (a,b): h -> a*h+b.
    # No division by tiny cumulative products; same recurrence as the loop.
    while offset < length:
        a_new = torch.cat([a[:, :offset], a[:, offset:] * a[:, :-offset]], dim=1)
        b_new = torch.cat(
            [b[:, :offset], b[:, offset:] + a[:, offset:] * b[:, :-offset]], dim=1
        )
        a, b = a_new, b_new
        offset *= 2
    y = (b * projected[:, :, r + n :].unsqueeze(-2)).sum(-1)
    return self.dropout(y + self.D * xc)


class FixedLinearCDE(nn.Module):
    """Exact default torchcde linear/RK4 step_size=1 convention, including
    left derivative at integer knots (the original uses perturb=False).
    """

    def __init__(self, vector_field, compiled=True):
        super().__init__()
        self.vector_field = vector_field
        self.step = (
            torch.compile(self.rk_step, fullgraph=True) if compiled else self.rk_step
        )

    def field(self, z, dx):
        return (
            self.vector_field.net(z)
            .view(-1, self.vector_field.d_hidden, self.vector_field.d_path)
            .matmul(dx.unsqueeze(-1))
            .squeeze(-1)
        )

    def rk_step(self, z, left_dx, dx):
        k1 = self.field(z, left_dx)
        k2 = self.field(z + k1 / 3, dx)
        k3 = self.field(z + k2 - k1 / 3, dx)
        k4 = self.field(z + k1 - k2 + k3, dx)
        return z + (k1 + 3 * (k2 + k3) + k4) / 8

    def forward(self, z0, path):
        increments = path[:, 1:] - path[:, :-1]
        states = [z0]
        for i in range(increments.shape[1]):
            states.append(
                self.step(states[-1], increments[:, max(0, i - 1)], increments[:, i])
            )
        return torch.stack(states, dim=1)


def accelerate(model, name, compiled=True):
    if name == "mamba":
        for layer in model.mamba_layers:
            layer.ssm.forward = types.MethodType(parallel_ssm, layer.ssm)
    elif name == "neural_cde":
        original = model.cde_solver
        assert (
            original.solver == "rk4"
            and original.step_size == 1
            and original.interpolation == "linear"
        )
        model.cde_solver = FixedLinearCDE(original.vector_field, compiled=compiled)
    elif name == "ode_rnn":
        model.step = (
            torch.compile(model.one_step, fullgraph=True)
            if compiled
            else model.one_step
        )
    return model
