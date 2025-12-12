import torch


def _zt_ellipse(t, a, b, w):
    # Position
    x = a * torch.cos(w * t)
    y = b * torch.sin(w * t)

    # 1st derivative (velocity)
    x_d = -a * w * torch.sin(w * t)
    y_d = b * w * torch.cos(w * t)

    # 2nd derivative (acceleration)
    x_dd = -a * w**2 * torch.cos(w * t)
    y_dd = -b * w**2 * torch.sin(w * t)

    # 3rd derivative (jerk)
    x_ddd = a * w**3 * torch.sin(w * t)
    y_ddd = -b * w**3 * torch.cos(w * t)

    # 4th derivative (snap)
    x_dddd = a * w**4 * torch.cos(w * t)
    y_dddd = b * w**4 * torch.sin(w * t)

    zt = torch.stack(
        [x, y, x_d, y_d, x_dd, y_dd, x_ddd, y_ddd, x_dddd, y_dddd], dim=-1
    )
    return zt


def _zt_lemniscate(t, a, b, w):
    theta = w * t
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    denominator = 1 + sin_theta**2

    # Position
    x = a * cos_theta / denominator
    y = b * sin_theta * cos_theta / denominator

    # 1st derivative (velocity)
    x_d = a * w * (sin_theta**2 - 3) * sin_theta / denominator**2
    y_d = b * w * (1 - 3 * sin_theta**2) / denominator**2

    # 2nd derivative (acceleration)
    x_dd = a * w**2 * (-sin_theta**4 + 12 * sin_theta**2 - 3) * cos_theta / denominator**3
    y_dd = 2 * b * w**2 * (14 * torch.sin(2 * theta) + 3 * torch.sin(4 * theta)) / (torch.cos(2 * theta) - 3)**3

    # 3rd derivative (jerk)
    x_ddd = a * w**3 * (-sin_theta**6 + 43 * sin_theta**4 - 103 * sin_theta**2 + 45) * sin_theta / denominator**4
    y_ddd = 4 * b * w**3 * (48 * sin_theta**6 - 328 * sin_theta**4 + 352 * sin_theta**2 - 40) / (torch.cos(2 * theta) - 3)**4

    # 4th derivative (snap)
    x_dddd = a * w**4 * (sin_theta**8 - 136 * sin_theta**6 + 730 * sin_theta**4 - 624 * sin_theta**2 + 45) * cos_theta / denominator**5
    y_dddd = 256 * b * w**4 * (3 * sin_theta**6 - 50 * sin_theta**4 + 107 * sin_theta**2 - 32) * sin_theta * cos_theta / (torch.cos(2 * theta) - 3)**5

    zt = torch.stack(
        [x, y, x_d, y_d, x_dd, y_dd, x_ddd, y_ddd, x_dddd, y_dddd], dim=-1
    )
    return zt


def generate_ellipse_reference(principle_axes, dt, Tf, angular_vel):
    """
    Generate a reference trajectory that is an ellipse in the x-y plane,
    including derivatives up to 4th order.

    Input:
        principle_axes: torch.tensor(2), lengths of the principal axes [a, b]
        dt: float, time step
        Tf: float, final time
        angular_vel: float, angular velocity

    Return:
        trajectory: torch.tensor(num_steps, 10), columns are:
            x, y, x_d, y_d, x_dd, y_dd, x_ddd, y_ddd, x_dddd, y_dddd
    """
    t = torch.arange(0, Tf, dt)
    a = principle_axes[0]
    b = principle_axes[1]
    w = angular_vel
    trajectory = _zt_ellipse(t, a, b, w)
    return trajectory


def generate_lemniscate_reference(principle_axes, dt, Tf, angular_vel):
    """
    Generate a reference trajectory that is a lemniscate in the x-y plane,
    including derivatives up to 4th order.

    Input:
        principle_axes: torch.tensor(2), lengths of the principal axes [a, b]
        dt: float, time step
        Tf: float, final time
        angular_vel: float, angular velocity

    Return:
        trajectory: torch.tensor(num_steps, 10), columns are:
            x, y, x_d, y_d, x_dd, y_dd, x_ddd, y_ddd, x_dddd, y_dddd
    """
    t = torch.arange(0, Tf, dt)
    a, b = principle_axes[0], principle_axes[1]
    w = angular_vel
    trajectory = _zt_lemniscate(t, a, b, w)
    return trajectory
