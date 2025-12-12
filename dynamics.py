import torch


class Dynamics:
    def __init__(self):
        pass

    def dot_fn(self, x, u):
        """Return the derivative of the state x given the input u"""
        raise NotImplementedError


class PlanarQuadDynamics(Dynamics):
    """Dynamics of the 2D quadrotor
    """

    def __init__(self, m_quad=1.0, I_quad=0.1, g=9.81, F_max=10.0, tau_max=0.5, **kwargs):
        super().__init__()
        self.m_quad = m_quad
        self.I_quad = I_quad
        self.g = g
        self.F_max = F_max
        self.tau_max = tau_max
        self.u_max = torch.tensor([F_max, tau_max])

    def dot_fn(self, x, u):
        u_saturated = torch.clamp(u, -self.u_max, self.u_max)
        x_dot = torch.vstack(
            [
                x[:, 2],
                x[:, 3],
                -u_saturated[:, 0] * torch.sin(x[:, 4]) / self.m_quad,
                u_saturated[:, 0] * torch.cos(x[:, 4]) / self.m_quad - self.g,
                torch.atan2(torch.sin(x[:, 5]), torch.cos(x[:, 5])),
                u_saturated[:, 1] / self.I_quad,
            ]
        ).T
        return x_dot, u_saturated


class PlanarQuadDynamicsWithDrag(PlanarQuadDynamics):
    """Dynamics of the 2D quadrotor with drag (squared air speed)"""

    def __init__(self, m_quad=1.0, I_quad=0.1, g=9.81, F_max=10.0, tau_max=0.5, C_pd=[0.1, 0.1], C_rdx=0.1, **kwargs):
        """Cd: drag coefficient"""
        super().__init__(m_quad, I_quad, g, F_max, tau_max)
        self.C_pd = torch.tensor(C_pd)
        self.C_rd = torch.tensor([C_rdx, 0])

    def drag(self, x):
        assert x.shape[-1] == 6, "Input should be a batch of states"
        drag_para = -self.C_pd * torch.linalg.vector_norm(x[..., 2:4], dim=-1, keepdim=True) * x[..., 2:4]
        drag_rotor = -self.C_rd * x[..., 2:4]
        return drag_para + drag_rotor

    def dot_fn(self, x, u):
        x_dot, u_saturated = super().dot_fn(x, u)    # nominal dynamics
        x_dot[:, 2:4] += self.drag(x)   # add drag
        return x_dot, u_saturated
