"""Test script for solving the linear shallow water equations."""

from shallow_water.solvers.parameters import LinearParameters
from shallow_water.solvers.linear import ExplicitSolver
import numpy as np
import matplotlib.pyplot as plt

# Parameters
grid = np.linspace(0, 1, 101)
dx = grid[1] - grid[0]
dt = 0.001
g = 9.81
theta = 0
rho = 1
H = 1
U = 1
initial_h = 1 + 0.2 * np.cos(2 * np.pi * grid)
initial_u = np.zeros_like(grid)

params = LinearParameters(dx=dx, dt=dt, g=g, theta=theta, rho=rho,
                          grid=grid, H=H, U=U, initial_h=initial_h, initial_u=initial_u)

# Solver
solver = ExplicitSolver(params)

# Time loop
fig = plt.figure()

for i in range(10):
    for j in range(20):
        solver.step()

    plt.plot(grid, solver.h, label=f"Time step {i}")

plt.legend()
plt.savefig("output.png")
