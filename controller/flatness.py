import numpy as np
import scipy.linalg as spl
import torch


def third_derivative(f, x):
    """
    Returns the 3rd derivative of f at x.
    If f: R^d -> R^m, we get a tensor of shape (m, d, d, d).
    """
    # We define a function for the jacobian, then take a jacobian of that again, etc.
    # first_jac_fn(x) -> shape (m, d)
    first_jac_fn = torch.func.jacrev(f)
    # second_jac_fn(x) -> shape (m, d, d)
    second_jac_fn = torch.func.jacrev(first_jac_fn)
    # third_jac_fn(x) -> shape (m, d, d, d)
    third_jac_fn = torch.func.jacrev(second_jac_fn)
    return third_jac_fn(x)


class FlatnessController:
    def __init__(self, model, reference, controller, observer, dt):
        self.model = model

        self.reference = reference
        self.dt = dt

        AB = spl.expm(dt * np.eye(5, k=1))
        self.A_flat = np.kron(AB, np.eye(2))

        self.controller = controller
        self.observer = observer
        self.z_estimate = None

        self.prev_u = None
        self.t = 0
        self.reset()

    def flat_step(self, z, v, dt=1e-2, integration="exact"):
        # z = [x1, x2, x1', x2', x1'', x2'', x1''', x2''']
        if integration == "euler":
            A = np.eye(8) + dt * np.eye(8, k=2)
            B = dt * np.eye(8)[:, -2:]
            return z @ A.T + v @ B.T
        else:
            # Exact computation of the stepping with matrix exponential
            return np.hstack([z, v]) @ self.A_flat[:8].T

    def compute_perturbations(self, z_np, v_np):
        raise NotImplementedError('compute_perturnbations() method has not been implemented!')

    def z2x_model(self, z, v):
        raise NotImplementedError('z2x_model() method has not been implemented!')

    def z2u_model(self, z, v, return_out=False):
        raise NotImplementedError('z2u_model() method has not been implemented!')

    def x2z(self, x, u):
        """
        Convert the state and control input to the flat state.
        Args:
            x: The current state of the system: [x, y, x_dot, y_dot, theta, theta_dot]
            u: The current control input
        Returns:
            The flat state of the system: [x, y, x_dot, y_dot, x_ddot, y_ddot, x_dddot, y_dddot]
        """
        raise NotImplementedError('x2z() method has not been implemented!')

    def control(self, state):
        raise NotImplementedError('control() method has not been implemented!')

    def reset(self):
        raise NotImplementedError('reset() method has not been implemented!')


class FlatQuadrotorController(FlatnessController):
    def __init__(self, model, quad_params, reference, controller, observer, dt):
        super().__init__(model, reference, controller, observer, dt)

        # Load quad params
        self.m_quad = quad_params['m_quad']
        self.I_quad = quad_params['I_quad']
        self.g = quad_params['g']
        self.F_max = quad_params['F_max']
        self.tau_max = quad_params['tau_max']

    def compute_perturbations(self, z_np, v_np):
        with torch.no_grad():
            if isinstance(z_np, torch.Tensor):
                z, v = z_np, v_np
            else:
                z, v = torch.tensor(z_np).float(), torch.tensor(v_np).float()
            in_dim, out_dim = 4, 6  # input and output dimensions of the model
            model_inputs = z[:,:4]
            out = self.model(model_inputs).numpy()
            batched_jacobian = torch.vmap(torch.func.jacrev(self.model))(model_inputs)
            batched_hessian = torch.vmap(torch.func.hessian(self.model))(model_inputs)
            input_dot = z[:, 2:6]
            input_ddot = z[:, 4:8]
            # First order derivatives
            out_dot = (
                torch.bmm(batched_jacobian, input_dot.unsqueeze(-1)).flatten(1).numpy()
            )
            # Second order derivatives
            batched_jac_dt = torch.bmm(
                batched_hessian.reshape(-1, in_dim, in_dim),
                input_dot.unsqueeze(1).repeat(1, out_dim, 1).reshape(-1, in_dim, 1),
            ).reshape(-1, out_dim, in_dim)
            out_ddot = (
                torch.bmm(batched_jacobian, input_ddot.unsqueeze(-1)).flatten(1).numpy()
                + torch.bmm(batched_jac_dt, input_dot.unsqueeze(-1)).flatten(1).numpy()
            )
        return out, out_dot, out_ddot

    def z2x_model(self, z, v):
        if self.model is not None:
            out, out_dot, _ = self.compute_perturbations(z, v)
            zpert = np.hstack([
                np.zeros_like(z[:, :4]),
                out[:, 2:4],
                out_dot[:, 2:4]
            ])
            ztilde = z - zpert
        else:
            ztilde = z
        x = np.vstack(
            [
                ztilde[:, 0],
                ztilde[:, 1],
                ztilde[:, 2],
                ztilde[:, 3],
                np.arctan2(-ztilde[:, 4], ztilde[:, 5] + self.g),
                (ztilde[:, 7] * ztilde[:, 4] - ztilde[:, 6] * (ztilde[:, 5] + self.g))
                / (ztilde[:, 4] ** 2 + (ztilde[:, 5] + self.g) ** 2)
            ]
        ).T
        if self.model is not None:
            x[:, 5] = x[:, 5] - out[:, 4]
        return x

    def z2u_model(self, z, v, return_out=False):
        EPS = 1e-6
        if self.model is not None:
            out, out_dot, out_ddot = self.compute_perturbations(z, v)
            zpert = np.hstack([
                np.zeros_like(z[:, :4]),
                out[:, 2:4],
                out_dot[:, 2:4]
            ])
            ztilde = z - zpert
            vtilde = v - out_ddot[:, 2:4]
        else:
            ztilde = z
            vtilde = v
        sos = ztilde[:, 4] ** 2 + (ztilde[:, 5] + self.g) ** 2
        u1 = self.m_quad * np.sqrt(sos)
        u2_denom = np.clip(sos**2, EPS, None)
        u2 = (
            self.I_quad
            * (
                (vtilde[:, 1] * ztilde[:, 4] - vtilde[:, 0] * (ztilde[:, 5] + self.g)) * sos
                + 2
                * (ztilde[:, 6] * (ztilde[:, 5] + self.g) - ztilde[:, 7] * ztilde[:, 4])
                * (ztilde[:, 6] * ztilde[:, 4] + ztilde[:, 7] * (ztilde[:, 5] + self.g))
            ) / u2_denom
        )
        if self.model is not None:
            u2 = u2 - out_dot[:, 4] - out[:, 5]

        if return_out:
            return np.vstack([u1, u2]).T, (out, out_dot, out_ddot)
        else:
            return np.vstack([u1, u2]).T

    def x2z(self, x, u):
        """
        Convert the state and control input to the flat state.
        Args:
            x: The current state of the system: [x, y, x_dot, y_dot, theta, theta_dot]
            u: The current control input
        Returns:
            The flat state of the system: [x, y, x_dot, y_dot, x_ddot, y_ddot, x_dddot, y_dddot]
        """
        z = np.concatenate((x[:4], np.zeros(4)))
        z[4] = -u[0] * np.sin(x[4]) / self.m_quad
        z[5] = u[0] * np.cos(x[4]) / self.m_quad - self.g
        # The line below is an approximate that assumes that \dot F = 0
        z[6:8] = np.array([-u[0] * np.cos(x[4]) * x[5], -u[0] * np.sin(x[4]) * x[5]]) / self.m_quad
        return z

    def control(self, state):
        # Even though we take in the state, we assume we can only observe the first four
        # values of the state vector. The rest are estimated.
        zt = self.z_estimate.copy()

        # Compute flat input with PD correction
        z_diff = zt - self.reference[self.t][:8]
        v_cmd = self.reference[self.t][8:] - self.controller.control(z_diff[:, None])[:, 0]

        # Compute nominal feedforward control
        if self.model is not None:
            u_cmd, outs = self.z2u_model(zt[None, :], v_cmd[None, :], return_out=True)
            u_cmd = u_cmd[0]
        else:
            u_cmd = self.z2u_model(zt[None, :], v_cmd[None, :])[0]
            outs = (np.zeros((1, 6)), np.zeros((1, 6)), np.zeros((1, 6)))

        u_cmd = np.clip(u_cmd, [0, -self.tau_max], [self.F_max, self.tau_max])
        # v_cmd = self.adjust_for_saturation(zt[None, :], v_cmd[None, :], u_cmd[None, :])[0]
        self.prev_u = u_cmd

        # Step observer for acceleration and jerk
        # self.z_estimate = self.observer.step(state[:4, None], v_cmd[:, None])[:, 0]
        obs = np.concatenate((
            state[:4],
            np.array([
                -u_cmd[0] * np.sin(state[4]) / self.m_quad,
                u_cmd[0] * np.cos(state[4]) / self.m_quad - self.g
            ] + outs[0][0, 2:4])
        ))
        self.z_estimate = self.observer.step(obs[:, None], v_cmd[:, None])[:, 0]

        self.t += 1
        return u_cmd

    def reset(self):
        self.z_estimate = self.reference[0, :8]
        if self.observer is not None:
            self.observer.x_hat = self.z_estimate[:, None]
        # self.z_estimate = np.concatenate((self.reference[0, :4], np.zeros(4)))
        self.t = 0
        self.prev_u = None


class FlatExampleController(FlatnessController):
    def __init__(self, model, reference, controller, observer, dt):
        super().__init__(model, reference, controller, observer, dt)

    def z2x_model(self, z, v):
        if self.model is not None:
            with torch.no_grad():
                zpert = self.model(z).numpy()
            ztilde = z - zpert
        else:
            ztilde = z
        x = np.vstack(
            [
                ztilde[:, 0] + np.sqrt(ztilde[:, 1]),
                np.sqrt(ztilde[:, 1])
            ]
        ).T
        return x

    def z2u_model(self, z, v, return_out=False):
        EPS = 1e-6
        if self.model is not None:
            with torch.no_grad():
                zpert = self.model(z).numpy()
            ztilde = z - zpert
        else:
            ztilde = z
        vtilde = v
        u = vtilde / (2 * np.sqrt(ztilde[:, 1]))
        return u

    def x2z(self, x, u):
        """
        Convert the state and control input to the flat state.
        Args:
            x: The current state of the system: [x1, x2]
            u: The current control input
        Returns:
            The flat state of the system: [x1 - x2, x2^2]
        """
        z = np.zeros_like(x)
        z[0] = x[0] - x[1]
        z[1] = np.square(x[1])
        return z

    def control(self, state):
        # For simplicity, we assume that we have perfect knowledge of the state in our observations
        # Compute flat input with PD correction
        z_diff = state - self.reference[self.t][:2]
        v_cmd = self.reference[self.t][2:] - self.controller.control(z_diff[:, None])[:, 0]

        # Compute nominal feedforward control
        u_cmd = self.z2u_model(state[None, :], v_cmd[None, :])[0]

        self.t += 1
        return u_cmd

    def reset(self):
        self.z_estimate = self.reference[0, :4]
        if self.observer is not None:
            self.observer.x_hat = self.z_estimate[:, None]
        self.t = 0
        self.prev_u = None
