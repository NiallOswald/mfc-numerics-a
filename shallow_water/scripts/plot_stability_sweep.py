#! /usr/bin/env python
from shallow_water.solvers.parameters import (LinearParameters,
                                              AnalyticParameters)
from shallow_water.solvers.linear import (ForwardBackwardSolver,
                                          AnalyticSolver)
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
from math import *  # noqa: F401,F403

INTERPOLATION_POINTS = 3
G = 9.81
U = 0.


def plot_stability_sweep():  # noqa: D103
    parser = ArgumentParser(
        description='Plot the stability of the forward-backward method for '
        'various step sizes.',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--start_point", type=float, default=0., help='The start point of the '
        'domain.'
    )
    parser.add_argument(
        "--end_point", type=float, default=1., help='The end point of the '
        'domain.'
    )
    parser.add_argument(
        "--final_time", type=float, default=1., help='The target final time '
        'of the simulation.'
    )
    parser.add_argument(
        "initial_h", type=str, nargs=1,
        help='An expression in the coordinate x for the inital condition of '
        'the variable h. The function should be a quoted string. E.g. '
        '"sin(x)". The function must be periodic and always return a float.'
    )
    parser.add_argument(
        "initial_u", type=str, nargs=1,
        help='An expression in the coordinate x for the inital condition of '
        'the variable u. The function should be a quoted string. E.g. '
        '"sin(x)". The function must be periodic and always return a float.'
    )
    parser.add_argument(
        "--theta", type=float, default=0., help='The angle of the bed.'
    )
    parser.add_argument(
        "--H", type=float, default=1., help='The reference depth of the fluid.'
    )
    parser.add_argument(
        "--grid_size", type=int, default=2**8, help='The grid size to use for '
        'the convergence study.'
    )
    parser.add_argument(
        "--beta_start", type=float, default=1., help='The starting value of '
        'beta to use for the stability study.'
    )
    parser.add_argument(
        "--beta_end", type=float, default=2., help='The ending value of beta '
        'to use for the stability study.'
    )
    parser.add_argument(
        "--beta_steps", type=int, default=20, help='The number of steps to '
        'take between the start and end values of beta.'
    )
    parser.add_argument(
        "--path", type=str, default='.', help='The path to save the plot.'
    )

    args = parser.parse_args()
    start_point = args.start_point
    end_point = args.end_point
    final_time = args.final_time
    initial_h = eval('lambda x: ' + args.initial_h[0])
    initial_u = eval('lambda x: ' + args.initial_u[0])
    theta = args.theta
    H = args.H  # noqa: N806
    grid_size = args.grid_size
    beta_start = args.beta_start
    beta_end = args.beta_end
    beta_steps = args.beta_steps
    path = args.path

    # Vectorize the initial conditions
    initial_h = np.vectorize(initial_h)
    initial_u = np.vectorize(initial_u)

    # Create the grid
    grid = np.linspace(start_point, end_point, grid_size + 1)
    dx = grid[1] - grid[0]

    # Set step sizes
    beta_values = np.linspace(beta_start, beta_end, beta_steps)
    dt_values = beta_values * dx / np.sqrt(G * H)

    errors = np.zeros((len(beta_values), 2))

    for i, dt in enumerate(dt_values):
        lin_params = LinearParameters(
            dt, G, theta, grid, H, U, initial_h(grid), initial_u(grid)
        )
        ana_params = AnalyticParameters(
            dt, G, theta, grid, H, U, initial_h, initial_u
        )

        solver = ForwardBackwardSolver(lin_params)
        exact = AnalyticSolver(ana_params)

        n_steps = int(final_time / dt)
        for _ in range(n_steps):
            solver.step()
            exact.step()

        h_error = np.linalg.norm(solver.h - exact.h, np.inf)
        u_error = np.linalg.norm(solver.u - exact.u, np.inf)
        print(f"beta = {beta_values[i]}, h_error = {h_error}, "
              f"u_error = {u_error}, final_time = {n_steps * dt}")

        errors[i] = [h_error, u_error]

    # Plot the errors
    plt.figure()
    plt.plot(beta_values, errors[:, 0], "k-", label=r"$h$")
    plt.plot(beta_values, errors[:, 1], "k--", label=r"$\bar{u}$")

    plt.ylim([np.min(errors) / 2, 1])
    plt.yscale("log")

    plt.xlabel(r"$\beta$")
    plt.ylabel("Error")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"{path}/stability_sweep.png", dpi=300)
