import torch
import torch.nn as nn


class FullResidualModel(nn.Module):
    """A fully connected neural network model for the residual 2d quad dynamics"""

    def __init__(self, hidden_dims=[16], activation=nn.GELU):
        super(FullResidualModel, self).__init__()
        state_dim, control_dim = 6, 2
        layers = []
        dims = [state_dim + control_dim] + hidden_dims + [state_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.model = nn.Sequential(*layers)
        self.y_mean = nn.Parameter(torch.zeros(6), requires_grad=False)
        self.y_std = nn.Parameter(torch.zeros(6), requires_grad=False)

    def forward(self, xu):
        out = self.model(xu)
        if self.y_mean is not None:
            out = out * self.y_std + self.y_mean
        return out

    def set_mean_std(self, y_mean, y_std):
        with torch.no_grad():
            self.y_mean[:] = y_mean
            self.y_std[:] = y_std


class FlatResidualModel(nn.Module):
    """A NN-parameterized flatness-preserving residual dynamics model.
    Can only accomodate disturbance as a function of position and velocity."""

    def __init__(self, hidden_dims=[16], activation=nn.GELU):
        super(FlatResidualModel, self).__init__()
        self.substate_dim = 2
        self.state_dim = 6
        num_substates = int(self.state_dim / self.substate_dim)
        self.subresiduals = nn.ModuleList()
        for i in range(num_substates):
            layers = []
            dims = [self.substate_dim * (i+1)] + hidden_dims + [self.substate_dim]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(activation())
            model = nn.Sequential(*layers)
            self.subresiduals.append(model)
        self.y_mean = nn.Parameter(torch.zeros(self.state_dim), requires_grad=False)
        self.y_std = nn.Parameter(torch.zeros(self.state_dim), requires_grad=False)

    def forward(self, xu):
        if xu.size(-1) == 4:
            xu = torch.cat([xu, torch.zeros_like(xu)], dim=-1)
        ret = []
        for i, m in enumerate(self.subresiduals):
            idx = 2 * (i + 1)
            ret.append(m(xu[..., :idx]))  # only position and velocity
        # Pad the first two entries with zeros
        ret = torch.cat(ret, dim=-1)
        if self.y_mean is not None:
            ret = ret * self.y_std + self.y_mean
        return ret

    def set_mean_std(self, y_mean, y_std):
        with torch.no_grad():
            self.y_mean[:] = y_mean
            self.y_std[:] = y_std


class HybridModel(nn.Module):
    """A closed-form nominal + NN residual dynamics model for the quadrotor.
    Considers an augmented state space with the control input as an additional dimension."""

    def __init__(self, dynamics, model):
        super(HybridModel, self).__init__()
        self.dynamics = dynamics
        self.model = model

    def forward(self, t, state_control):
        """
        Args:
            - t:                float, time
            - state_control:    torch.tensor(num_agents, state_dim + control_dim),
                                state-control pairs
        """
        x, u = state_control[:, :-2], state_control[:, -2:]
        dxdt = self.dynamics.dot_fn(x, u)[0] + self.model(state_control)
        dudt = torch.zeros_like(u)
        return torch.cat([dxdt, dudt], dim=-1)
    