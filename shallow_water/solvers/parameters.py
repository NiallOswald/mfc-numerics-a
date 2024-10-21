"""Parameters for the Saint-Venant equation."""

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class NonlinearParameters:
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


@dataclass
class LinearParameters:
    """Parameters for the linearized Saint-Venant equation."""

    dx: float
    dt: float
    g: float
    theta: float
    rho: float
    grid: np.ndarray
    H: float
    U: float
    initial_h: np.ndarray
    initial_u: np.ndarray


@dataclass
class AnalyticParameters:
    """Parameters for the linearized Saint-Venant equation."""

    dt: float
    g: float
    theta: float
    grid: np.ndarray
    H: float
    U: float
    initial_h: Callable[[float], float]
    initial_u: Callable[[float], float]
