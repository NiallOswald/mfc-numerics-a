"""Solvers for the linearized Saint-Venant equation."""

from .parameters import Parameters
from abc import ABC, abstractmethod
import numpy as np


class Solver(ABC, object):
    """Base class for solvers of the linearized Saint-Venant equation."""

    def __init__(self, params: Parameters):
        """Initialize the solver.

        :param params: The :class:`Parameters` used for the solver.
        """
        self.params = params
        self.dt = params.dt
        self.x = params.grid
        self.h = params.initial_h(self.x)
        self.u = params.initial_u(self.x)

    @abstractmethod
    def step(self):
        """Perform a single time step."""
        pass


class ForwardSolver(Solver):
    """Forward-in-time solver for the linearized Saint-Venant equation."""

    def step(self):
        """Perform a single time step."""
        new_h = np.zeros_like(self.h)
        new_u = np.zeros_like(self.u)

        # Update the interior points
        new_h[1:-1] = self.h[1:-1] - self.dt / (2 * self.params.dx) * (
            self.params.H * (self.u[2:] - self.u[:-2])
            + self.params.U * (self.h[2:] - self.h[:-2])
        )
        new_u[1:-1] = (
            self.u[1:-1]
            - self.dt
            / (2 * self.params.dx)
            * (
                self.params.U * (self.u[2:] - self.u[:-2])
                + self.params.g * (self.h[2:] - self.h[:-2])
            )
            + self.dt * self.params.g * self.params.theta
        )

        # Update the boundary points
        new_h[0] = self.h[0] - self.dt / (2 * self.params.dx) * (
            self.params.H * (self.u[1] - self.u[-2])
            + self.params.U * (self.h[1] - self.h[-2])
        )
        new_u[0] = (
            self.u[0]
            - self.dt
            / (2 * self.params.dx)
            * (
                self.params.U * (self.u[1] - self.u[-2])
                + self.params.g * (self.h[1] - self.h[-2])
            )
            + self.dt * self.params.g * self.params.theta
        )

        new_h[-1] = new_h[0]
        new_u[-1] = new_u[0]

        self.h = new_h
        self.u = new_u


class ForwardBackwardSolver(Solver):
    """Fowards-backwards solver for the linearized Saint-Venant equation."""

    def step(self):
        """Perform a single time step."""
        new_h = np.zeros_like(self.h)
        new_u = np.zeros_like(self.u)

        # Forwards step
        new_h[1:-1] = self.h[1:-1] - self.dt / (2 * self.params.dx) * (
            self.params.H * (self.u[2:] - self.u[:-2])
        )
        new_h[0] = self.h[0] - self.dt / (2 * self.params.dx) * (
            self.params.H * (self.u[1] - self.u[-2])
        )
        new_h[-1] = new_h[0]

        # Backwards step
        new_u[1:-1] = (
            self.u[1:-1]
            - self.dt
            / (2 * self.params.dx)
            * (self.params.g * (new_h[2:] - new_h[:-2]))
            + self.dt * self.params.g * self.params.theta
        )
        new_u[0] = (
            self.u[0]
            - self.dt / (2 * self.params.dx) * (self.params.g * (new_h[1] - new_h[-2]))
            + self.dt * self.params.g * self.params.theta
        )
        new_u[-1] = new_u[0]

        self.h = new_h
        self.u = new_u


class AnalyticSolver(Solver):
    """Analytic solver for the linearized Saint-Venant equation."""

    def __init__(self, params: Parameters):
        """Initialize the solver.

        :param params: The :class:`Parameters` used for the solver.
        """
        self.params = params
        self.dt = self.params.dt

        self.solution = self._generate_solution()
        self.time = 0

    def _generate_solution(self):
        """Compute the analytic solution."""
        # Setup
        A = np.array(  # noqa: N806
            [[self.params.U, self.params.H], [self.params.g, self.params.U]]
        )
        eigs, eigv = np.linalg.eig(A)

        init = lambda x: np.array([self.params.initial_h(x), self.params.initial_u(x)])
        forcing = np.array([0, self.params.g * self.params.theta])

        # Compute the solution in eigenspace
        inv_init = lambda x: np.linalg.solve(eigv, init(x))
        inv_forcing = np.linalg.solve(eigv, forcing)
        inv_sol = lambda x, t: (
            inv_forcing[:, np.newaxis] * t
            + np.array(
                [inv_init(x - eigs[0] * t)[0, :], inv_init(x - eigs[1] * t)[1, :]]
            )
        )

        # Invert the transformation
        sol = lambda x, t: eigv @ inv_sol(x, t)

        return sol

    def evaluate(self, x: float, t: float):
        """Evaluate the analytic solution at time `t`."""
        return self.solution(x, t)

    def __call__(self, t: float):
        """Evaluate the analytic solution at time `t` on the grid."""
        return self.evaluate(self.params.grid, t)

    def step(self):
        """Simulate a time-step for comparison with numerical solutions."""
        self.time += self.dt

    @property
    def h(self):  # noqa: D102
        return self(self.time)[0, :]

    @property
    def u(self):  # noqa: D102
        return self(self.time)[1, :]
