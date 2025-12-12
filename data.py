import torch
from utils.sim_utils import rk4_step


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
