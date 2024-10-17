"""Parameters for the Saint-Venant equation."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Parameters:
    """Parameters for the Saint-Venant equation."""

    dx: float
    dt: float
    g: float
    theta: float
    tau: float
    rho: float
    grid: np.ndarray
    initial_h: np.ndarray
    initial_u: np.ndarray
