import numpy as np
import torch
import time


def rk4_step(f, x, u, dt):
    """
    Perform a single Runge-Kutta 4th order (RK4) step.

    Parameters:
        f (function): The function defining the dynamics, f(x, u).
        x (torch tensor): Current state, x(t).
        u (torch tensor): Control input, u(t).
        dt (float): Time step size.

    Returns:
        torch tensor: Updated state, x(t + dt).
    """
    k1, u_sat = f(x, u)
    k2, _ = f(x + dt / 2 * k1, u)
    k3, _ = f(x + dt / 2 * k2, u)
    k4, _ = f(x + dt * k3, u)

    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next, u_sat


def rk4_step_time_varying(f, x, dt, t):
    """
    Perform a single Runge-Kutta 4th order (RK4) step.

    Parameters:
        f (function): The function defining the dynamics, f(x, u).
        x (torch tensor): Current state, x(t).
        dt (float): Time step size.
        t (float): Current time.

    Returns:
        torch tensor: Updated state, x(t + dt).
    """
    k1 = f(x, t)
    k2 = f(x + dt / 2 * k1, t + dt / 2)
    k3 = f(x + dt / 2 * k2, t + dt / 2)
    k4 = f(x + dt * k3, t + dt)

    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


def simulate_autonomous(dynamics, dt, num_steps, x0, seed=None):
    """
    Simulate the system in open-loop without control inputs.

    Args:
        dynamics (function): The system dynamics.
        dt (float): Time step.
        num_steps (int): Number of steps to simulate.
        x0 (torch tensor): Initial state.
        seed (int): Random seed.

    Returns:
        torch tensor: State trajectories.
    """
    if seed is not None:
        torch.manual_seed(seed)
    xtrajectories = torch.zeros((num_steps + 1, x0.shape[0], x0.shape[1]))
    xtrajectories[0] = x0
    for t in range(num_steps):
        xtrajectories[t + 1] = rk4_step(dynamics, xtrajectories[t], torch.zeros_like(x0), dt)[0]
    return xtrajectories


def simulate_time_varying(dynamics, dt, num_steps, x0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    xtrajectories = torch.zeros((num_steps + 1, x0.shape[0], x0.shape[1]))
    xtrajectories[0] = x0
    for n in range(num_steps):
        xtrajectories[n + 1] = rk4_step_time_varying(
            dynamics, xtrajectories[n], dt, n * dt
        )[0]
    return xtrajectories


def simulate_openloop(dynamics, dt, num_steps, x0, us, obs_noise_std=0.0, u_noise_std=0.0, seed=None):
    """Simulate the system in open-loop
    Args:
        dynamics: function, the system dynamics
        dt: float, time step
        num_steps: int, number of steps to simulate
        x0: torch.tensor(batch_size, state_dim), initial state
        us: torch.tensor(batch_size, num_steps, control_dim), control inputs
        obs_noise_std: float or 6-dim tensor, standard deviation of the observation noise
        u_noise_std: float or 2-dim tensor, standard deviation of the control input noise
        seed: int, random seed
    Returns:
        xtrajectories: torch.tensor(batch_size, num_steps+1, state_dim), state trajectories
        utrajectories: torch.tensor(batch_size, num_steps, control_dim), control input trajectories
    """
    assert (
        x0.shape[0] == us.shape[0]
    ), "initial state and control input should have the same batch size"
    assert (
        us.shape[1] == num_steps
    ), "control input array should have the same number of steps as num_steps"
    xtrajectories = torch.zeros((x0.shape[0], num_steps + 1, x0.shape[1]))
    utrajectories = torch.zeros((x0.shape[0], num_steps, us.shape[2]))
    xtrajectories[:, 0, :] = x0
    for t in range(num_steps):
        obs = xtrajectories[:, t, :] + torch.randn_like(xtrajectories[:, t, :]) * obs_noise_std
        u_noisy = us[:, t, :] + torch.randn_like(us[:, t, :]) * u_noise_std
        xtrajectories[:, t + 1, :], u_sat = rk4_step(dynamics, obs, u_noisy, dt)
        utrajectories[:, t, :] = u_sat
    return xtrajectories, utrajectories


def simulate_closedloop(dynamics, num_samples, dt, Tf, x0, controller, obs_noise_std=0.0, u_noise_std=0.0, seed=None, sampling_period=1, measure_time=False):
    assert num_samples == 1, "Closed-loop simulation only supports num_samples=1"
    controller.reset()
    if seed is not None:
        torch.manual_seed(seed)
    num_steps = int(Tf / dt)
    xtrajectories = torch.zeros((num_samples, num_steps + 1, 6))
    utrajectories = torch.zeros((num_samples, num_steps, 2))
    xtrajectories[:, 0, :] = x0
    if measure_time:
        ts = []
    for t in range(num_steps):
        # Gather observation
        obs = xtrajectories[0, t, :].clone()
        obs += torch.randn_like(obs) * obs_noise_std
        # Update the control action
        if t % sampling_period == 0:
            if measure_time:
                start = time.time()
            u = torch.tensor(controller.control(obs.numpy())).unsqueeze(0)
            if measure_time:
                ts.append(time.time() - start)
        # Step dynamics
        u_noisy = u + torch.randn_like(u) * u_noise_std
        xtrajectories[:, t + 1, :], u_sat = rk4_step(dynamics, xtrajectories[:, t, :], u_noisy, dt)
        utrajectories[:, t, :] = u_sat
    if measure_time:
        return xtrajectories, utrajectories, np.array(ts)
    else:
        return xtrajectories, utrajectories


def sample_random_trajectories(
    dynamics, num_samples, dt, Tf, x0_range, u_range, obs_noise_std=0.0, u_noise_std=0.0, seed=None
):
    """Sample initial states and control inputs to generate trajectories"""
    if seed is not None:
        torch.manual_seed(seed)
    num_steps = int(Tf / dt)
    x0 = torch.rand(num_samples, 6) * (x0_range[1] - x0_range[0]) + x0_range[0]
    us = torch.rand((num_samples, num_steps, 2)) * (u_range[1] - u_range[0]) + u_range[0]
    xtraj, utraj = simulate_openloop(dynamics, dt, num_steps, x0, us, obs_noise_std, u_noise_std)
    return xtraj, utraj
