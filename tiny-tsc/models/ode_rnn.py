"""ODE-RNN with Euler evolution and an observation update at every time point."""

import torch
from torch import nn


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


def create_net(n_inputs, n_outputs, n_layers=1, n_units=100, nonlinear=nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)


class GRU_Unit(nn.Module):
    def __init__(self, latent_dim, input_dim, n_units=100):
        super(GRU_Unit, self).__init__()

        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid(),
        )
        init_network_weights(self.update_gate)

        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid(),
        )
        init_network_weights(self.reset_gate)

        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim * 2),
        )
        init_network_weights(self.new_state_net)

    def forward(self, y_mean, y_std, x, mask):
        y_concat = torch.cat([y_mean, y_std, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)

        new_state, new_state_std = torch.chunk(
            self.new_state_net(concat), chunks=2, dim=-1
        )
        new_state_std = new_state_std.abs()

        output_y = (1 - update_gate) * new_state + update_gate * y_mean
        new_y_std = (1 - update_gate) * new_state_std + update_gate * y_std

        mask = (torch.sum(mask, -1, keepdim=True) > 0).float()

        new_y = mask * output_y + (1 - mask) * y_mean
        new_y_std = mask * new_y_std + (1 - mask) * y_std

        new_y_std = new_y_std.abs()
        return output_y, new_y, new_y_std


class ODERNN(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.channels, self.latent_dim = channels, 32
        flow = create_net(32, 32, n_layers=1, n_units=100, nonlinear=nn.Tanh)
        init_network_weights(flow)
        self.ode_solver = nn.Module()
        self.ode_solver.ode_func = nn.Module()
        self.ode_solver.ode_func.gradient_net = flow
        self.gru_unit = GRU_Unit(32, channels, n_units=100)
        self.classifier = nn.Sequential(
            nn.Linear(32, 100), nn.ReLU(), nn.Dropout(0.1), nn.Linear(100, 2)
        )
        init_network_weights(self.classifier)
        self.step = self.one_step

    def one_step(self, h, std, previous, value, mask, dt):
        h = h + self.ode_solver.ode_func.gradient_net(h) * dt
        observation = previous * (1 - mask) + value * mask
        _, h, std = self.gru_unit(h, std, observation, mask)
        return h, std, observation

    def forward(self, x, times):
        h = x.new_zeros((len(x), self.latent_dim))
        std = torch.zeros_like(h)
        previous = x.new_zeros((len(x), self.channels))
        for j in range(x.shape[1]):
            dt = (times[:, j] - times[:, max(0, j - 1)]).unsqueeze(-1)
            h, std, previous = self.step(
                h, std, previous, x[:, j, : self.channels], x[:, j, self.channels :], dt
            )
        return self.classifier(h)
