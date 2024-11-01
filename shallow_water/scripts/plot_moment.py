#! /usr/bin/env python
from shallow_water.parameters import Parameters
from shallow_water.solvers import ForwardSolver, ForwardBackwardSolver, AnalyticSolver
from shallow_water.utils import func_from_str, parameter_parser, path_parser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 12


def plot_moment():  # noqa: D103
    parser = ArgumentParser(
        description="Plot the moments of the variables for the linear shallow "
        "water equations.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--final_time", type=float, default=1.0, help="The time to plot up to."
    )
    parser.add_argument(
        "moment",
        type=str,
        nargs=1,
        help="An expression in the variables h and u for the moment to compute. The "
        'function should be a quoted string. E.g. "u**2 + h**2".',
    )
    parser.add_argument(
        "--grid_size", type=int, default=2**8, help="The grid size to use."
    )
    parser.add_argument(
        "--method",
        choices=["forward", "forward_backward"],
        default="forward_backward",
        help='The method to use for the simulation. Selecting "forward_backward" will '
        "override the U parameter.",
    )
    parser.add_argument(
        "--plot_error", action="store_true", help="Plot the error in the moment."
    )
    parser.add_argument(
        "--y_label", type=str, default="Moment", help="The label for the y-axis."
    )
    parameter_parser(parser)
    path_parser(parser)
    args = parser.parse_args()

    # Process the parameters
    kwargs = vars(args)

    final_time = args.final_time
    method_str = args.method
    plot_error = args.plot_error
    y_label = args.y_label
    path = args.path
    tag = args.tag

    moment = func_from_str(args.moment[0], vars="h, u")
    kwargs["initial_h"] = func_from_str(args.initial_h[0])
    kwargs["initial_u"] = func_from_str(args.initial_u[0])
    kwargs["dt"] = func_from_str(args.dt, vars="dx, U, H, g")

    # Set the parameters
    params = Parameters.from_dict(**kwargs)

    # Select the method
    if method_str == "forward":
        solver_class = ForwardSolver
    elif method_str == "forward_backward":
        solver_class = ForwardBackwardSolver
        kwargs["U"] = 0.0
    else:
        raise ValueError(f"Unknown method {method_str}")

    # Setup solvers
    solver = solver_class(params)
    exact = AnalyticSolver(params)

    n_steps = int(final_time / params.dt)
    moment_values = np.zeros(n_steps)

    for i in range(n_steps):
        # Perform a single time step
        solver.step()
        exact.step()

        # Compute the moment
        integrand = moment(solver.h, solver.u)
        moment_values[i] = np.trapz(integrand, params.grid) / (
            params.end_point - params.start_point
        )

        # Compute the error (if required)
        if plot_error:
            exact_integrand = moment(exact.h, exact.u)
            exact_moment = np.trapz(exact_integrand, params.grid) / (
                params.end_point - params.start_point
            )
            moment_values[i] -= exact_moment
            moment_values[i] = abs(moment_values[i])

    # Plot the moment
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(n_steps) * params.dt, moment_values, "k-")

    plt.xlabel(r"$t$")
    plt.ylabel("Error" if plot_error else y_label)

    if plot_error:
        plt.yscale("log")

    plt.tight_layout()

    plt.savefig(f"{path}moment{tag}.png", dpi=300)
