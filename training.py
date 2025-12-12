import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchdiffeq import odeint
from utils.sim_utils import rk4_step


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
        input_dim, output_dim = 2, 4
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.model = nn.Sequential(*layers)
        self.y_mean = nn.Parameter(torch.zeros(6), requires_grad=False)
        self.y_std = nn.Parameter(torch.zeros(6), requires_grad=False)

    def forward(self, xu):
        ret = self.model(xu[..., 2:4])  # only position and velocity
        # Pad the first two entries with zeros
        ret = torch.cat([torch.zeros_like(xu[..., :2]), ret], dim=-1)
        # ret[..., :2] = 0
        # ret[..., 4:] = 0
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


class ResidualDataset(Dataset):
    def __init__(self, xtrajs, utrajs, dynamics_fn, dt):
        self.dynamics_fn = dynamics_fn
        self.dt = dt
        residual = self.finite_diff_residual(xtrajs, utrajs)
        x = torch.cat([xtrajs[:, :-1], utrajs], dim=-1)
        self.x = x.reshape(-1, x.shape[-1]).float()

        # Normalize the residual
        self.residual = residual.reshape(-1, residual.shape[-1]).float()
        self.residual_mean = self.residual.mean(dim=0)
        self.residual_std = self.residual.std(dim=0) + 1e-6
        # self.residual = (residual - self.residual_mean) / self.residual_std

    def finite_diff_residual(self, xtraj, utraj):
        """Approximate the residual of the unicycle dynamics with finite difference
        x is a trajectory of shape (num_samples, num_steps + 1, Nx)
        u is a trajectory of shape (num_samples, num_steps, Nu)
        """
        num_samples, num_steps, _ = utraj.shape
        res = torch.zeros_like(xtraj[:, :-1])
        for t in range(num_steps):
            pred_next = rk4_step(self.dynamics_fn, xtraj[:, t], utraj[:, t], self.dt)[0]
            res[:, t] = (xtraj[:, t + 1] - pred_next) / self.dt
        return res

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.residual[idx]


def get_data_loaders(dataset, train_frac=0.8, batch_size=128, num_workers=4):
    num_train = int(train_frac * len(dataset))
    num_val = len(dataset) - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def train_finite_difference(
    xtraj, utraj, dt, dynamics_fn, model, lr=1e-3, epochs=100, batch_size=128, device="cpu"
):
    """Learn the perturbation function using a neural network
    Input:
        - xtraj: torch.Tensor of shape (num_samples, num_steps + 1, Nx), the state trajectory
        - utraj: torch.Tensor of shape (num_samples, num_steps, Nu), the control trajectory
        - dt: float, time step
        - dynamics: function, nominal dynamics of the system
        - model: nn.Module, the neural network model
        - lr: float, learning rate
        - epochs: int, number of epochs to train the network
        - batch_size: int, batch size
    Output:
        - model: nn.Module, the learned perturbation function
    """
    # Compute residual from data; construct data
    residual_dataset = ResidualDataset(xtraj, utraj, dynamics_fn, dt)
    model.set_mean_std(residual_dataset.residual_mean, residual_dataset.residual_std)
    train_loader, val_loader = get_data_loaders(residual_dataset, batch_size=batch_size)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-6)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, residual_batch in train_loader:
            x_batch, residual_batch = x_batch.to(device), residual_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            train_loss = criterion(output, residual_batch)
            train_loss.backward()
            optimizer.step()
            train_loss += train_loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, residual_val in val_loader:
                x_val, residual_val = x_val.to(device), residual_val.to(device)
                output = model(x_val)
                val_loss += criterion(output, residual_val).item()
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Validation Loss: {val_loss/len(val_loader):.4f}",
            end='\r'
        )
    model.eval().to('cpu')
    return model


###################################################################################################
# Neural ODE Training: needs further tuning, thus not used in the final project
###################################################################################################
def train_NODE(
    hybrid_model, state_traj, control_traj, time_stamps, lr=1e-3, epochs=100, dt=1, device="cpu"
):
    """
    Train the Neural ODE model using the given state and control trajectories.

    Args:
        - hybrid_model:   HybridModel instance to train.
        - state_traj:     torch.Tensor, shape [batch_size, num_time_steps, state_dim].
                          Ground truth state trajectory.
        - control_traj:   torch.Tensor, shape [batch_size, num_time_steps, control_dim].
                          Control inputs for each trajectory.
        - time_stamps:    torch.Tensor, shape [num_time_steps], time stamps for integration.
        - lr:             float, learning rate.
        - epochs:         int, number of epochs for training.
        - dt:             float, time step for integration, used here to scale the loss
    """
    hybrid_model.train().to(device)
    optimizer = optim.Adam(hybrid_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    state_traj, control_traj = state_traj.to(device), control_traj.to(device)
    time_stamps = time_stamps.to(device)

    batch_size, num_time_steps, state_dim = state_traj.shape
    _, _, control_dim = control_traj.shape

    for epoch in range(epochs):
        epoch_loss = 0.0
        x_pred = state_traj[:, 0]  # Initialize x_pred to true initial state
        for t in range(num_time_steps - 2):
            xt = state_traj[:, t]
            z0 = torch.cat([xt, control_traj[:, t]], dim=-1)
            t_span = time_stamps[t: t + 2]
            z_pred = odeint(hybrid_model, z0, t_span, method="rk4")[1]
            x_pred, x_true = z_pred[:, :-control_dim], state_traj[:, t + 1]
            loss = loss_fn(x_pred, x_true) / dt  # Scale loss by dt

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Average loss for the epoch
        epoch_loss /= num_time_steps - 1
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}", end='\r')
    hybrid_model.eval().to('cpu')
