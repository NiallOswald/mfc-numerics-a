#! /usr/bin/env python
from shallow_water.solvers.parameters import (LinearParameters,
                                              AnalyticParameters)
from shallow_water.solvers.linear import UpwindSolver, AnalyticSolver
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
from math import *  # noqa: F401,F403

G = 9.81


def plot_linear_convergence():  # noqa: D103
    parser = ArgumentParser(
        description='Plot the convergence of the linear shallow water '
        'equations.',
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
        "--final_time", type=float, default=1., help='The final time of the '
        'simulation.'
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
        "--U", type=float, default=1., help='The reference velocity of the '
        'fluid.'
    )

    args = parser.parse_args()
    start_point = args.start_point
    end_point = args.end_point
    final_time = args.final_time
    initial_h = eval('lambda x: ' + args.initial_h[0])
    initial_u = eval('lambda x: ' + args.initial_u[0])
    theta = args.theta
    H = args.H  # noqa: N806
    U = args.U  # noqa: N806

    # Vectorize the initial conditions
    initial_h = np.vectorize(initial_h)
    initial_u = np.vectorize(initial_u)

    n_values = 2 ** np.arange(4, 12)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']

    for i, n in enumerate(n_values):
        grid = np.linspace(start_point, end_point, n)

        # Choose dt such that the stability condition is satisfied
        dx = grid[1] - grid[0]
        dt = 0.5 * min(dx / (U + G), dx / (H + U))

        lin_params = LinearParameters(
            dt, G, theta, grid, H, U, initial_h(grid), initial_u(grid)
        )
        ana_params = AnalyticParameters(
            dt, G, theta, grid, H, U, initial_h, initial_u
        )

        solver = UpwindSolver(lin_params)
        exact = AnalyticSolver(ana_params)

        n_steps = int(final_time / dt)
        for _ in range(n_steps):
            solver.step()
            exact.step()

        h_error = np.linalg.norm(solver.h - exact.h, np.inf)
        u_error = np.linalg.norm(solver.u - exact.u, np.inf)
        print(f"n = {n}, h_error = {h_error}, u_error = {u_error}, "
              f"final_time = {n_steps * dt}")

        plt.plot(grid, solver.h, label=f"n = {n}", color=colors[i])
        plt.plot(grid, exact.h, color=colors[i], linestyle='--')

    plt.legend()
    plt.savefig("output.png")
