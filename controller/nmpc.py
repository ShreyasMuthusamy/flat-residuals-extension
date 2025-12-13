import casadi as cs
import numpy as np
from torch import nn


def relu(x: cs.MX) -> cs.MX:
    return cs.fmax(0, x)


def gelu(x: cs.MX, approximate: str = "none") -> cs.MX:
    if approximate == "none":
        return 0.5 * x * (1 + cs.erf(x / cs.sqrt(2)))
    x3 = cs.power(x, 3)
    return 0.5 * x * (1 + cs.tanh(cs.sqrt(2 / cs.pi) * (x + 0.044715 * x3)))


def torch_nn_2_casadi(model: nn.Module, y_mean: np.ndarray, y_std: np.ndarray) -> cs.Function:
    """Convert a PyTorch neural network to a CasADi function."""
    assert isinstance(model, nn.Module), "Input must be a PyTorch nn.Module"

    param_ls = []
    for param in model.state_dict().values():
        param_ls.append(param.detach().cpu().numpy())
    x_dim = param_ls[0].shape[1]
    assert x_dim == 8 or x_dim == 2, (
        "We currently only support full (x, u) or flat (y) parameterization for the "
        "residual dynamics"
    )

    x = cs.MX.sym("x", x_dim)
    out = x
    param_idx = 0
    for layer in model:
        if isinstance(layer, nn.Linear):
            weight, bias = param_ls[param_idx], param_ls[param_idx + 1]
            out = cs.mtimes(weight, out) + bias
            param_idx += 2
        elif isinstance(layer, nn.Tanh):
            out = cs.tanh(out)
        elif isinstance(layer, nn.ReLU):
            out = relu(out)
        elif isinstance(layer, nn.GELU):
            out = gelu(out)
        else:
            raise NotImplementedError(f"Unsupported layer type: {type(layer)}")
    if out.shape[0] == 4:
        out = cs.vertcat(cs.MX.zeros(2, 1), out)
    if y_mean is not None:
        out = out * y_std + y_mean
    return cs.Function("nn_casadi", [x], [out]), x_dim


class PlanarQuadrotorMPC:
    def __init__(
        self,
        residual_model,     # ResidualModel
        quad_params: dict,
        dt: float,
        reference: np.ndarray,
        horizon: int = 20,
    ):
        """
        Args:
            neural_net:     ResidualModel, neural network model for the residual dynamics
            m_quad:         float, mass of the quadrotor
            I_quad:         float, moment of inertia of the quadrotor
            g:              float, acceleration due to gravity
            dt:             float, time step used for the MPC prediction
            reference:      np.ndarray, reference trajectory for the quadrotor; **shape (4, T)**
            horizon:        int, length of the MPC horizon
        """

        self.m_quad = quad_params["m_quad"]
        self.I_quad = quad_params["I_quad"]
        self.g = quad_params["g"]
        self.F_max = quad_params["F_max"]
        self.tau_max = quad_params["tau_max"]
        self.horizon = horizon
        self.prediction_dt = dt
        if residual_model is None:
            self.nn_casadi = None
        else:
            self.nn_casadi, self.nn_in_dim = torch_nn_2_casadi(
                residual_model.model, residual_model.y_mean.numpy(), residual_model.y_std.numpy()
            )
        self._set_dynamics()
        self.init_mpc = 0
        self.prev_sol_x = np.zeros((1,))
        self.prev_lam_g = np.zeros((1,))
        self.reference = reference
        assert self.reference.shape[0] == 4, "Invalid reference shape, reference should be 4xT"
        self.t = 0

        self._setup_opti()

    def _setup_opti(self):
        """Create and store the optimization problem."""
        self.opti = cs.Opti()

        # **Define variables**
        self.x = self.opti.variable(6, self.horizon + 1)  # State trajectory
        self.u = self.opti.variable(2, self.horizon)  # Control inputs

        # **Define a parameter for the initial state**
        self.x0 = self.opti.parameter(6)
        self.current_ref = self.opti.parameter(4, self.horizon + 1)

        # **Define the cost function**
        self.cost = 0
        for k in range(self.horizon):
            # Note: this cost is slightly different from the one in the paper (as it also penalizes velocity error
            # instead of just position error). Just penalizing position error leads to an unstable controller in
            # the case where there is observation and control noise.
            self.cost += cs.sumsqr(self.x[:4, k] - self.current_ref[:4, k]) + 1e-3 * cs.sumsqr(self.u[:, k])
            self.opti.subject_to(self.x[:, k + 1] == self.dynamics(self.x[:, k], self.u[:, k]))

        self.opti.minimize(self.cost)

        # **Define constraints**
        self.opti.subject_to(self.opti.bounded(0, self.u[0, :], self.F_max))
        self.opti.subject_to(self.opti.bounded(-self.tau_max, self.u[1, :], self.tau_max))
        self.opti.subject_to(self.x[:, 0] == self.x0)  # Use parameter instead of direct constraint

        # **Solver settings**
        self.opti.solver(
            "ipopt",
            dict(expand=True, print_time=False),
            dict(print_level=0, warm_start_init_point="yes"),
        )

    def _set_dynamics(self):
        x = cs.MX.sym("x", 6, 1)
        u = cs.MX.sym("u", 2, 1)
        nominal_dynamics = cs.vertcat(
            x[2],
            x[3],
            -u[0] * cs.sin(x[4]) / self.m_quad,
            u[0] * cs.cos(x[4]) / self.m_quad - self.g,
            x[5],
            u[1] / self.I_quad,
        )
        if self.nn_casadi is not None:
            if self.nn_in_dim == 8:
                residual_dynamics = self.nn_casadi(cs.vertcat(x, u))
            elif self.nn_in_dim == 2:
                residual_dynamics = self.nn_casadi(x[2:4])
            hybrid_dynamics = nominal_dynamics + residual_dynamics
        else:
            hybrid_dynamics = nominal_dynamics
        f = cs.Function("f", [x, u], [hybrid_dynamics])
        integrator = cs.integrator(
            "integrator",
            "rk",
            {"x": x, "p": u, "ode": f(x, u)},
            {"tf": self.prediction_dt, "number_of_finite_elements": 4, "simplify": True},
        )
        result = integrator(x0=x, p=u)
        self.dynamics = cs.Function("Dynamics", [x, u], [result["xf"]])
        return 1

    def control(self, state, return_full=False):
        # Find the reference traj for this step; extend the reference if needed
        reference = self.reference[:, self.t: self.t + self.horizon + 1]
        if reference.shape[1] < self.horizon + 1:
            if reference.shape[1] == 0:
                return np.zeros(2)
            reference = np.hstack(
                [reference, np.tile(reference[:, -1:], (1, self.horizon + 1 - reference.shape[1]))]
            )

        self.opti.set_value(self.x0, state)
        self.opti.set_value(self.current_ref, reference)

        if self.init_mpc >= 1:
            self.opti.set_initial(self.opti.x, self.prev_sol_x)
            self.opti.set_initial(self.opti.lam_g, self.prev_lam_g)

        # Solve
        sol = self.opti.solve()

        # **Store solution for warm-starting next iteration**
        self.init_mpc += 1
        self.prev_sol_x = sol.value(self.opti.x)
        self.prev_lam_g = sol.value(self.opti.lam_g)
        self.t += 1

        if return_full:
            return sol.value(self.u)
        else:
            return sol.value(self.u[:, 0])

    def control_recompile(self, state):
        """
        Args:
            state: np.ndarray, current state of the quadrotor; **shape (6,)**
        Returns:
            u: np.ndarray, control input for the quadrotor; **shape (2,)**
        """
        # Find the reference traj for this step; extend the reference if needed
        reference = self.reference[:, self.t: self.t + self.horizon + 1]
        if reference.shape[1] < self.horizon + 1:
            reference = np.hstack(
                [reference, np.tile(reference[:, -1:], (1, self.horizon + 1 - reference.shape[1]))]
            )

        opti = cs.Opti()
        x = opti.variable(6, self.horizon + 1)
        u = opti.variable(2, self.horizon)

        cost = 0
        for k in range(self.horizon):
            cost += cs.sumsqr(x[:4, k] - reference[:4, k]) + 1e-3 * cs.sumsqr(u[:, k])
            opti.subject_to(x[:, k + 1] == self.dynamics(x[:, k], u[:, k]))

        opti.minimize(cost)

        opti.subject_to(opti.bounded(0, u[0, :], self.F_max))
        opti.subject_to(opti.bounded(-self.tau_max, u[1, :], self.tau_max))
        opti.subject_to(x[:, 0] == state)

        opti.solver(
            "sqpmethod",
            dict(expand=True, print_time=False),
            dict(expand=True, print_level=0, warm_start_init_point="yes"),
        )

        if self.init_mpc >= 1:
            opti.set_initial(opti.x, self.prev_sol_x)
            opti.set_initial(opti.lam_g, self.prev_lam_g)

        sol = opti.solve()

        self.init_mpc += 1
        self.prev_sol_x = sol.value(opti.x)
        self.prev_lam_g = sol.value(opti.lam_g)
        self.t += 1

        return sol.value(u[:, 0])

    def reset(self):
        self.t = 0
