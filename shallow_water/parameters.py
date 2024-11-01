"""Parameters for the Saint-Venant equation."""

from dataclasses import dataclass
import inspect
from typing import Callable
import numpy as np


@dataclass
class Parameters:
    """Parameters for the linearized Saint-Venant equation."""

    theta: float
    H: float
    initial_h: Callable[[float], float]
    initial_u: Callable[[float], float]
    dt: Callable[[float, float, float, float], float]
    start_point: float
    end_point: float
    grid_size: int
    g: float = 9.81
    U: float = 0.0

    def __post_init__(self):  # noqa: D105
        """Setup post-initialization."""  # noqa: D401
        # Create the grid
        self.grid = np.linspace(self.start_point, self.end_point, self.grid_size + 1)
        self.dx = self.grid[1] - self.grid[0]

        # Setup the initial conditions
        self.initial_h = np.vectorize(self.initial_h)
        self.initial_u = np.vectorize(self.initial_u)

        # Setup the time-step
        self.dt = self.dt(self.dx, self.U, self.H, self.g)

    @classmethod
    def from_dict(cls, **kwargs):
        """Create a :class:`Parameters` from a dictionary."""
        return cls(
            **{
                k: v
                for k, v in kwargs.items()
                if k in inspect.signature(cls).parameters
            }
        )
