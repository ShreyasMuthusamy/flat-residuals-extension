import numpy as np
import scipy.signal as ss
import cvxpy as cp


class LuenbergerObserver:
    def __init__(self, A, B, C, poles, x0_hat=None):
        """
        Initialize the Luenberger Observer.

        Parameters
        ----------
        A : np.ndarray
            State transition matrix (n x n).
        B : np.ndarray
            Control input matrix (n x m).
        C : np.ndarray
            Observation matrix (p x n).
        poles : np.ndarray
            Desired poles of the observer (n x 1).
        """
        obs = ss.place_poles(A.T, C.T, poles.flatten())
        L = obs.gain_matrix.T
        self.A = np.kron(A, np.eye(2))
        self.B = np.kron(B, np.eye(2))
        self.C = np.kron(C, np.eye(2))
        self.L = np.kron(L, np.eye(2))

        if x0_hat is None:
            self.x_hat = np.zeros((self.A.shape[0], 1))
        else:
            if x0_hat.ndim == 1:
                x0_hat = x0_hat[:, None]
            self.x_hat = x0_hat

    def step(self, y, u):
        """
        Perform one full observer iteration.

        Parameters
        ----------
        u : np.ndarray
            Control input at time k.
        y : np.ndarray
            Measurement at time k.

        Returns
        -------
        np.ndarray
            The updated state estimate, x_hat(k+1).
        """
        # x^(k+1) = A x(k) + B u(k) + L (y(k) - C x(k))
        y_residual = y - self.C @ self.x_hat
        self.x_hat = self.A @ self.x_hat + self.B @ u + self.L @ y_residual

        # Additional information: y-acceleration is bounded; scale everything else accordingly
        # thrust_used = self.z2u_model(self.z_estimate[None, :], v[None, :])[0, 0]
        # if thrust_used > self.F_max:
        #     self.z_estimate[4] = self.z_estimate[4] / thrust_used * self.F_max
        #     self.z_estimate[5] = (self.z_estimate[5] + self.g) / thrust_used * self.F_max - self.g
        #     self.z_estimate[6:8] = self.z_estimate[6:8] / thrust_used * self.F_max
        return self.x_hat


class KalmanFilter:
    def __init__(self, A, B, C, Q, R,
                 x0=None, P0=None):
        """
        Initialize the discrete-time Kalman Filter.

        Parameters
        ----------
        A : np.ndarray
            State transition matrix (n x n).
        B : np.ndarray
            Control input matrix (n x m).
        C : np.ndarray
            Observation matrix (p x n).
        Q : np.ndarray
            Process noise covariance (n x n).
        R : np.ndarray
            Measurement noise covariance (p x p).
        x0 : np.ndarray, optional
            Initial state estimate (n x 1). Defaults to zero.
        P0 : np.ndarray, optional
            Initial estimate covariance (n x n). Defaults to identity.
        """
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R

        n = A.shape[0]  # state dimension
        if x0 is None:
            self.x_hat = np.zeros((n, 1))
        else:
            self.x_hat = x0

        if P0 is None:
            self.P = np.eye(n)
        else:
            self.P = P0

        self.K = None  # Kalman Gain will be computed each update

    def predict(self, u):
        """
        Perform the prediction step of the Kalman Filter.

        Parameters
        ----------
        u : np.ndarray
            Control input at time k (m x 1).
        """
        # x^-(k+1) = A x(k) + B u(k)
        self.x_hat = self.A @ self.x_hat + self.B @ u

        # P^-(k+1) = A P(k) A^T + Q
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):
        """
        Perform the update step of the Kalman Filter with the new measurement.

        Parameters
        ----------
        y : np.ndarray
            Measurement at time k+1 (p x 1).
        """
        # S = C P^- C^T + R
        S = self.C @ self.P @ self.C.T + self.R

        # K(k+1) = P^- C^T [S]^-1
        self.K = self.P @ self.C.T @ np.linalg.inv(S)

        # x^(k+1) = x^-(k+1) + K(k+1) (y(k+1) - C x^-(k+1))
        y_residual = y - self.C @ self.x_hat
        self.x_hat = self.x_hat + self.K @ y_residual

        # P(k+1) = (I - K(k+1) C) P^-
        I = np.eye(self.P.shape[0])
        self.P = (I - self.K @ self.C) @ self.P

    def step(self, y, u):
        """
        Perform one full Kalman filter iteration (predict + update).

        Parameters
        ----------
        u : np.ndarray
            Control input at time k.
        y : np.ndarray
            Measurement at time k+1.

        Returns
        -------
        np.ndarray
            The updated state estimate, x_hat(k+1).
        """
        self.predict(u)

        # Additional information: y-acceleration is bounded;
        # max_thrust_acc = 10.0
        # g = 9.81
        # self.x_hat[4] = np.clip(self.x_hat[4], -max_thrust_acc, max_thrust_acc)
        # self.x_hat[5] = np.clip(self.x_hat[5], -max_thrust_acc - g, max_thrust_acc - g)

        self.update(y)

        return self.x_hat


class LinearController:
    def __init__(self, A, B, poles):
        """
        Initialize the linear state feedback controller.

        Parameters
        ----------
        A : np.ndarray
            State transition matrix (n x n).
        B : np.ndarray
            Control input matrix (n x m).
        poles : np.ndarray
            Desired poles of the closed-loop system (n x 1).
        """
        controller = ss.place_poles(A, B, poles.flatten())
        K = controller.gain_matrix
        self.A = np.kron(A, np.eye(2))
        self.B = np.kron(B, np.eye(2))
        self.K = np.kron(K, np.eye(2))

    def control(self, x):
        """
        Compute the control input for the given state.

        Parameters
        ----------
        x : np.ndarray
            Current state (n x 1).
        """
        return self.K @ x


class Quad2DFlatMPC:
    def __init__(self, A, B, Q, R, quad_params, horizon):
        """
        Initialize the linear MPC controller.

        Parameters
        ----------
        A : np.ndarray
            State transition matrix (n x n).
        B : np.ndarray
            Control input matrix (n x m).
        Q : np.ndarray
            State cost matrix (n x n).
        R : np.ndarray
            Control input cost matrix (m x m).
        x_min : np.ndarray
            Minimum state constraints (n x 1).
        x_max : np.ndarray
            Maximum state constraints (n x 1).
        u_min : np.ndarray
            Minimum control input constraints (m x 1).
        u_max : np.ndarray
            Maximum control input constraints (m x 1).
        horizon : int
            MPC horizon (number of steps to look ahead).
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        self.m_quad = quad_params['m_quad']
        self.I_quad = quad_params['I_quad']
        self.g = quad_params['g']
        self.F_max = quad_params['F_max']
        self.tau_max = quad_params['tau_max']

        self.horizon = horizon

        self.n = A.shape[0]  # State dimension
        self.m = B.shape[1]  # Control dimension

        self._setup_opt_problem()

    def _setup_opt_problem(self):
        self.x0 = cp.Parameter(self.n)

        # Define optimization variables:
        #   X: states over the horizon (size n x (horizon+1))
        #   U: inputs over the horizon (size m x horizon)
        X = cp.Variable((self.n, self.horizon + 1))
        U = cp.Variable((self.m, self.horizon))

        # Define the cost function and constraints
        cost = 0
        constraints = []

        # Initial condition
        constraints.append(X[:, 0] == self.x0)

        # Build the cost and constraints over the horizon
        for t in range(self.horizon):
            # Cost accumulation: state cost + input cost
            cost += cp.quad_form(X[:, t], self.Q) + cp.quad_form(U[:, t], self.R)

            # System dynamics constraint: x_{t+1} = A*x_t + B*u_t
            constraints.append(X[:, t+1] == self.A @ X[:, t] + self.B @ U[:, t])

            # State constraints: (x[4] ** 2 + (x[5] + g) ** 2) ** 0.5 <= F_max / m_quad
            constraints.append(cp.norm(X[4:6, t], 2) <= self.F_max / self.m_quad)

            # TODO: Input constraints: v leads to feasible tau
            pass

        # Formulate the optimization problem
        self.problem = cp.Problem(cp.Minimize(cost), constraints)
        self.U = U

    def control(self, x0):
        """
        Compute the MPC control input for the given current state by solving
        a finite-horizon quadratic cost problem subject to linear dynamics
        and state/input constraints.

        Parameters
        ----------
        x0 : np.ndarray
            Current state (n x 1).

        Returns
        -------
        u0 : np.ndarray
            First-step control input (m x 1) for the given state.
        """

        # Solve the problem
        self.x0.value = x0[:, 0]
        self.problem.solve(warm_start=True)

        # Extract the first control input
        u0 = self.U.value[:, :1]  # This is the action to apply at the current step
        return u0
