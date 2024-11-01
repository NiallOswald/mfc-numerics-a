#! /usr/bin/env python
from shallow_water.parameters import Parameters
from shallow_water.solvers import ForwardSolver, ForwardBackwardSolver, AnalyticSolver
from shallow_water.utils import func_from_str, parameter_parser, path_parser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 12


def plot_comparison():  # noqa: D103
    parser = ArgumentParser(
        description="Plot a comparison of the convergence of the two schemes.",
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
        "--interpolation_points",
        type=int,
        default=3,
        help="The number of points to use for the line of best fit.",
    )
    parameter_parser(parser, allow_U=False)
    path_parser(parser)
    args = parser.parse_args()

    # Process the parameters
    kwargs = vars(args)

    final_time = args.final_time
    min_grid = args.min_grid
    max_grid = args.max_grid
    interpolation_points = args.interpolation_points
    path = args.path
    tag = args.tag

    kwargs["initial_h"] = func_from_str(args.initial_h[0])
    kwargs["initial_u"] = func_from_str(args.initial_u[0])
    kwargs["dt"] = func_from_str(args.dt, vars="dx, U, H, g")

    # Setup names
    names = ["Forward", "Forward-backward"]

    n_values = 2 ** np.arange(min_grid, max_grid + 1)
    errors = np.zeros((len(n_values), 2, 2))

    for i, n in enumerate(n_values):
        params = Parameters.from_dict(**kwargs, grid_size=n)

        solvers = [ForwardSolver(params), ForwardBackwardSolver(params)]
        exact_solver = AnalyticSolver(params)

        n_steps = int(final_time / params.dt)
        for _ in range(n_steps):
            for solver in solvers:
                solver.step()
            exact_solver.step()

        for j, (name, solver) in enumerate(zip(names, solvers)):
            h_error = np.linalg.norm(solver.h - exact_solver.h, np.inf)
            u_error = np.linalg.norm(solver.u - exact_solver.u, np.inf)
            print(
                f"n = {n}, {name.lower()} h_error = {h_error}, "
                f"{name.lower()} u_error = {u_error}, "
                f"final_time = {n_steps * params.dt}"
            )

            errors[i, j, :] = [h_error, u_error]

    # Plot the errors
    plt.figure(figsize=(8, 6))

    cmap = plt.get_cmap("tab10")

    for j, name in enumerate(names):
        plt.loglog(
            n_values,
            errors[:, j, 0],
            label=rf"{name} $h$",
            color=cmap(j),
            linestyle="-",
        )
        plt.loglog(
            n_values,
            errors[:, j, 1],
            label=f"{name} " + r"$\bar{u}$",
            color=cmap(j),
            linestyle="--",
        )

    # Plot a line of best fit
    m, c = np.polyfit(
        np.log(n_values[-interpolation_points:]),
        np.log(np.mean(errors[-interpolation_points:, :], axis=(1, 2))),
        1,
    )
    plt.loglog(n_values, np.exp(c) * n_values**m, "k:", label=r"$N^{" f"{m:.2f}" r"}$")

    plt.xlabel(r"$N$")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{path}convergence_comparison{tag}.png", dpi=300)
