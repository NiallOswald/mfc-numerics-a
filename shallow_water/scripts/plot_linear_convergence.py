#! /usr/bin/env python
from shallow_water.solvers.parameters import (LinearParameters,
                                              AnalyticParameters)
from shallow_water.solvers.linear import (ForwardSolver, ForwardBackwardSolver,
                                          AnalyticSolver)
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
from math import *  # noqa: F401,F403

INTERPOLATION_POINTS = 3
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
        "--U", type=float, default=1., help='The reference velocity of the '
        'fluid.'
    )
    parser.add_argument(
        "--max_grid", type=int, default=10, help='The maximum grid size to '
        'use for the convergence study.'
    )
    parser.add_argument(
        "--method", choices=['forward', 'forward_backward'],
        default='forward_backward', help='The method to use for the '
        'simulation. Selecting "forward_backward" will override the U '
        'parameter.'
    )
    parser.add_argument(
        "--plot_solutions", action='store_true', help='Plot the solutions.'
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
    U = args.U  # noqa: N806
    max_grid = args.max_grid
    method_str = args.method
    plot_solutions = args.plot_solutions
    path = args.path

    # Vectorize the initial conditions
    initial_h = np.vectorize(initial_h)
    initial_u = np.vectorize(initial_u)

    # Select the method
    if method_str == 'forward':
        solver_class = ForwardSolver
    elif method_str == 'forward_backward':
        solver_class = ForwardBackwardSolver
        U = 0.  # noqa: N806
    else:
        raise ValueError(f"Unknown method {method_str}")

    n_values = 2 ** np.arange(4, max_grid + 1)
    errors = np.zeros((len(n_values), 2))

    # Setup figure
    if plot_solutions:
        plt.figure()
        plt.plot([], [], "k-", label="Numerical")
        plt.plot([], [], "k--", label="Exact")

        colors = plt.cm.jet(np.linspace(0, 1, len(n_values)))

    for i, n in enumerate(n_values):
        grid = np.linspace(start_point, end_point, n + 1)

        # Choose dt such that the stability condition is satisfied
        dx = grid[1] - grid[0]
        dt = dx / np.sqrt(G * H)

        lin_params = LinearParameters(
            dt, G, theta, grid, H, U, initial_h(grid), initial_u(grid)
        )
        ana_params = AnalyticParameters(
            dt, G, theta, grid, H, U, initial_h, initial_u
        )

        solver = solver_class(lin_params)
        exact = AnalyticSolver(ana_params)

        n_steps = int(final_time / dt)
        for _ in range(n_steps):
            solver.step()
            exact.step()

        h_error = np.linalg.norm(solver.h - exact.h, np.inf)
        u_error = np.linalg.norm(solver.u - exact.u, np.inf)
        print(f"n = {n}, h_error = {h_error}, u_error = {u_error}, "
              f"final_time = {n_steps * dt}")

        errors[i] = [h_error, u_error]

        if plot_solutions:
            plt.plot(grid, solver.h, color=colors[i], linestyle="-")
            plt.plot(grid, exact.h, color=colors[i], linestyle="--")

    if plot_solutions:
        plt.xlabel(r"$x$")
        plt.ylabel(r"$h$")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"{path}/solutions.png", dpi=300)

    # Plot the errors
    plt.figure()
    plt.loglog(n_values, errors[:, 0], "k-", label=r"$h$")
    plt.loglog(n_values, errors[:, 1], "k--", label=r"$\bar{u}$")

    # Plot a line of best fit
    m, c = np.polyfit(
        np.log(n_values[-INTERPOLATION_POINTS:]),
        np.log(np.mean(errors[-INTERPOLATION_POINTS:, :], axis=1)),
        1,
    )
    plt.loglog(n_values, np.exp(c) * n_values**m, "k:",
               label=r"$N^{"f"{m:.2f}""}$")

    plt.xlabel(r"$N$")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{path}/linear_convergence.png", dpi=300)
