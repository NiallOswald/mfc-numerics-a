"""Solvers for the Saint-Venant equation."""

from .parameters import NonlinearParameters
import numpy as np


class ExplicitSolver(object):
    def __init__(self, params: NonlinearParameters):
        """Explicit solver for the Saint-Venant equation.

        :param params: The :class:`NonlinearParameters` used for the solver.
        """
        self.params = params
        self.x = params.grid
        self.h = params.initial_h
        self.u = params.initial_u

    def step(self):
        """Perform a single time step."""
        new_h = np.zeros_like(self.h)
        new_u = np.zeros_like(self.u)

        # Update the interior points
        new_h[1:-1] = self.h[1:-1] - self.params.dt / (2 * self.params.dx) * (
            self.h[1:-1] * (self.u[2:] - self.u[:-2]) + self.u[1:-1] * (self.h[2:] - self.h[:-2])
        )
        new_u[1:-1] = self.u[1:-1] - self.params.dt / (2 * self.params.dx) * (
            self.u[1:-1] * (self.u[2:] - self.u[:-2])
            + self.params.g * np.cos(self.params.theta) * (self.h[2:] - self.h[:-2])
        ) + self.params.dt * self.params.g * np.sin(self.params.theta) - self.params.dt * self.params.tau / (self.params.rho * self.h[1:-1])

        # Update the boundary points
        new_h[0] = self.h[0] - self.params.dt / (2 * self.params.dx) * (
            self.h[0] * (self.u[1] - self.u[-2]) + self.u[0] * (self.h[1] - self.h[-2])
        )
        new_u[0] = self.u[0] - self.params.dt / (2 * self.params.dx) * (
            self.u[0] * (self.u[1] - self.u[-2])
            + self.params.g * np.cos(self.params.theta) * (self.h[1] - self.h[-2])
        ) + self.params.dt * self.params.g * np.sin(self.params.theta) - self.params.dt * self.params.tau / (self.params.rho * self.h[0])

        new_h[-1] = new_h[0]
        new_u[-1] = new_u[0]

        self.h = new_h
        self.u = new_u
