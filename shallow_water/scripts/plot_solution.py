#! /usr/bin/env python
from shallow_water.parameters import Parameters
from shallow_water.solvers import ForwardSolver, ForwardBackwardSolver, AnalyticSolver
from shallow_water.utils import func_from_str, parameter_parser, path_parser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 12


def plot_solution():  # noqa: D103
    parser = ArgumentParser(
        description="Plot the solutions to the linear shallow water equations.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parameter_parser(parser)
    parser.add_argument(
        "--time_values",
        nargs="*",
        default=[1.0],
        help="The times at which to plot the solution.",
    )
    parser.add_argument(
        "--grid_size", type=int, default=2**8, help="The grid size to use."
    )
    parser.add_argument(
        "--method",
        choices=["forward", "forward_backward"],
        default="forward_backward",
        help='The method to use for the simulation. Selecting "forward_backward" '
        "will override the U parameter.",
    )
    parser.add_argument(
        "--variable", choices=["h", "u"], default="h", help="The variable to plot."
    )
    path_parser(parser)
    args = parser.parse_args()

    # Process the parameters
    kwargs = vars(args)

    time_values = args.time_values
    method_str = args.method
    variable_name = args.variable
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

    # Setup parameters
    params = Parameters.from_dict(**kwargs)

    # Setup time values
    time_values = [float(t) for t in time_values]
    cmap = plt.get_cmap("tab10")

    # Set the variable to plot
    if variable_name == "u":
        variable_name = r"\bar{u}"

    # Setup the plot
    plt.figure(figsize=(8, 6))
    plt.plot([], [], color="black", linestyle="--", label="Numerical")
    plt.plot([], [], color="black", linestyle="-", label="Analytic")

    for i, t in enumerate(time_values):
        num_solver = solver_class(params)
        exact_solver = AnalyticSolver(params)

        n_steps = int(t / params.dt)
        for _ in range(n_steps):
            num_solver.step()
            exact_solver.step()

        if variable_name == "h":
            plt.plot(
                params.grid,
                num_solver.h,
                color=cmap(i),
                linestyle="--",
            )
            plt.plot(
                params.grid,
                exact_solver.h,
                color=cmap(i),
                linestyle="-",
                label=f"t = {t}",
            )
        else:
            plt.plot(
                params.grid,
                num_solver.u,
                color=cmap(i),
                linestyle="--",
            )
            plt.plot(
                params.grid,
                exact_solver.u,
                color=cmap(i),
                linestyle="-",
                label=f"t = {t}",
            )

    plt.xlabel(r"$x$")
    plt.ylabel(rf"${variable_name}$")
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(f"{path}solutions{tag}.png", dpi=300)
