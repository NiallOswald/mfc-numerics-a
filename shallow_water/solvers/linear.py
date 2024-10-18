"""Solvers for the linearized Saint-Venant equation."""

from .parameters import LinearParameters
import numpy as np


class ExplicitSolver(object):
    def __init__(self, params: LinearParameters):
        """Explicit solver for the linearized Saint-Venant equation.

        :param params: The :class:`LinearParameters` used for the solver.
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
            self.params.H * (self.u[2:] - self.u[:-2]) + self.params.U * (self.h[2:] - self.h[:-2])
        )
        new_u[1:-1] = self.u[1:-1] - self.params.dt / (2 * self.params.dx) * (
            self.params.U * (self.u[2:] - self.u[:-2])
            + self.params.g * (self.h[2:] - self.h[:-2])
        ) + self.params.dt * self.params.g * self.params.theta

        # Update the boundary points
        new_h[0] = self.h[0] - self.params.dt / (2 * self.params.dx) * (
            self.params.H * (self.u[1] - self.u[-2]) + self.params.U * (self.h[1] - self.h[-2])
        )
        new_u[0] = self.u[0] - self.params.dt / (2 * self.params.dx) * (
            self.params.U * (self.u[1] - self.u[-2])
            + self.params.g * (self.h[1] - self.h[-2])
        ) + self.params.dt * self.params.g * self.params.theta

        new_h[-1] = new_h[0]
        new_u[-1] = new_u[0]

        self.h = new_h
        self.u = new_u
