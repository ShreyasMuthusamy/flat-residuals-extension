import torch
from torch.utils.data import DataLoader, Dataset, random_split
from utils.sim_utils import rk4_step
from utils.sim_utils import rk4_step


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


def sample_trajectories(
    dynamics,
    num_samples,
    dt,
    Tf,
    x0=None,
    us=None,
    x0_range=None,
    u_range=None,
    seed=None,
):
    """Sample initial states and control inputs to generate trajectories"""
    assert ((x0 is not None) and (us is not None)) or (
        (x0_range is not None) and (u_range is not None)
    ), "Either x0 or x0_range and u_range must be provided"
    if seed is not None:
        torch.manual_seed(seed)
    num_steps = int(Tf / dt)
    xtrajectories = torch.zeros((num_samples, num_steps + 1, 6))
    utrajectories = torch.zeros((num_samples, num_steps, 2))

    if x0 is not None:
        assert x0.shape[0] == num_samples, "x0 should have num_samples rows"
    else:
        x0 = torch.rand(num_samples, 6) * (x0_range[1] - x0_range[0]) + x0_range[0]
    xtrajectories[:, 0, :] = x0
    for t in range(num_steps):
        if us is not None:
            assert us.shape[0] == num_samples, "u should have num_samples rows"
            u = us[:, t, :]
        else:
            u = torch.rand(num_samples, 2) * (u_range[1] - u_range[0]) + u_range[0]
        xtrajectories[:, t + 1, :] = rk4_step(dynamics, xtrajectories[:, t, :], u, dt)
        utrajectories[:, t, :] = u
    return xtrajectories, utrajectories


def sample_trajectories_closed_loop(dynamics, num_samples, dt, Tf, x0, controller, seed=None):
    assert num_samples == 1, "Closed-loop simulation only supports num_samples=1"
    controller.reset()
    # if seed is not None:
    #     torch.manual_seed(seed)
    num_steps = int(Tf / dt)
    xtrajectories = torch.zeros((num_samples, num_steps + 1, 6))
    utrajectories = torch.zeros((num_samples, num_steps, 2))
    xtrajectories[:, 0, :] = x0
    for t in range(num_steps):
        u = torch.tensor(controller.control(xtrajectories[0, t, :].numpy())).unsqueeze(0)
        xtrajectories[:, t + 1, :] = rk4_step(dynamics, xtrajectories[:, t, :], u, dt)
        utrajectories[:, t, :] = u
    return xtrajectories, utrajectories
