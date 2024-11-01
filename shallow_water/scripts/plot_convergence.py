#! /usr/bin/env python
from shallow_water.parameters import Parameters
from shallow_water.solvers import ForwardSolver, ForwardBackwardSolver, AnalyticSolver
from shallow_water.utils import func_from_str, parameter_parser, path_parser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 12


def plot_convergence():  # noqa: D103
    parser = ArgumentParser(
        description="Plot the convergence of the linear shallow water equations.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--final_time",
        type=float,
        default=1.0,
        help="The target final time of the simulation.",
    )
    parser.add_argument(
        "--min_grid",
        type=int,
        default=4,
        help="The minimum grid size to use for the convergence study.",
    )
    parser.add_argument(
        "--max_grid",
        type=int,
        default=10,
        help="The maximum grid size to use for the convergence study.",
    )
    parser.add_argument(
        "--method",
        choices=["forward", "forward_backward"],
        default="forward_backward",
        help='The method to use for the simulation. Selecting "forward_backward" '
        "will override the U parameter.",
    )
    parser.add_argument(
        "--plot_solutions", action="store_true", help="Plot the solutions."
    )
    parser.add_argument(
        "--interpolation_points",
        type=int,
        default=3,
        help="The number of points to use for the line of best fit.",
    )
    parameter_parser(parser)
    path_parser(parser)
    args = parser.parse_args()

    # Process the parameters
    kwargs = vars(args)

    final_time = args.final_time
    min_grid = args.min_grid
    max_grid = args.max_grid
    method_str = args.method
    plot_solutions = args.plot_solutions
    interpolation_points = args.interpolation_points
    path = args.path
    tag = args.tag

    kwargs["initial_h"] = func_from_str(args.initial_h[0])
    kwargs["initial_u"] = func_from_str(args.initial_u[0])
    kwargs["dt"] = func_from_str(args.dt, vars="dx, U, H, g")

    # Select the method
    if method_str == "forward":
        solver_class = ForwardSolver
    elif method_str == "forward_backward":
        solver_class = ForwardBackwardSolver
        kwargs["U"] = 0.0
    else:
        raise ValueError(f"Unknown method {method_str}")

    n_values = 2 ** np.arange(min_grid, max_grid + 1)
    errors = np.zeros((len(n_values), 2))

    # Setup figure
    if plot_solutions:
        plt.figure(figsize=(8, 6))
        plt.plot([], [], "k--", label="Numerical")
        plt.plot([], [], "k-", label="Exact")

        cmap = plt.get_cmap("tab10")

    for i, n in enumerate(n_values):
        params = Parameters.from_dict(**kwargs, grid_size=n)

        num_solver = solver_class(params)
        exact_solver = AnalyticSolver(params)

        n_steps = int(final_time / params.dt)
        for _ in range(n_steps):
            num_solver.step()
            exact_solver.step()

        h_error = np.linalg.norm(num_solver.h - exact_solver.h, np.inf)
        u_error = np.linalg.norm(num_solver.u - exact_solver.u, np.inf)
        print(
            f"n = {n}, h_error = {h_error}, u_error = {u_error}, "
            f"final_time = {n_steps * params.dt}"
        )

        errors[i] = [h_error, u_error]

        if plot_solutions:
            plt.plot(params.grid, num_solver.h, color=cmap(i), linestyle="--")
            plt.plot(params.grid, exact_solver.h, color=cmap(i), linestyle="-")

    if plot_solutions:
        plt.xlabel(r"$x$")
        plt.ylabel(r"$h$")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"{path}convergence_solutions{tag}.png", dpi=300)

    # Plot the errors
    plt.figure(figsize=(8, 6))
    plt.loglog(n_values, errors[:, 0], "k-", label=r"$h$")
    plt.loglog(n_values, errors[:, 1], "k--", label=r"$\bar{u}$")

    # Plot a line of best fit
    m, c = np.polyfit(
        np.log(n_values[-interpolation_points:]),
        np.log(np.mean(errors[-interpolation_points:, :], axis=1)),
        1,
    )
    plt.loglog(n_values, np.exp(c) * n_values**m, "k:", label=r"$N^{" f"{m:.2f}" r"}$")

    plt.xlabel(r"$N$")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{path}linear_convergence{tag}.png", dpi=300)
