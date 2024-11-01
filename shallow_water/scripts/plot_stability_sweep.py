#! /usr/bin/env python
from shallow_water.parameters import Parameters
from shallow_water.solvers import ForwardBackwardSolver, AnalyticSolver
from shallow_water.utils import func_from_str, parameter_parser, path_parser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 12


def plot_stability_sweep():  # noqa: D103
    parser = ArgumentParser(
        description="Plot the stability of the forward-backward method for "
        "various step sizes.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--final_time",
        type=float,
        default=1.0,
        help="The target final time of the simulation.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=2**8,
        help="The grid size to use for the convergence study.",
    )
    parser.add_argument(
        "--beta_start",
        type=float,
        default=1.0,
        help="The starting value of beta to use for the stability study.",
    )
    parser.add_argument(
        "--beta_end",
        type=float,
        default=2.0,
        help="The ending value of beta to use for the stability study.",
    )
    parser.add_argument(
        "--beta_steps",
        type=int,
        default=20,
        help="The number of steps to take between the start and end values of beta.",
    )
    parameter_parser(parser, allow_U=False, allow_dt=False)
    path_parser(parser)
    args = parser.parse_args()

    # Process the parameters
    kwargs = vars(args)

    final_time = args.final_time
    beta_start = args.beta_start
    beta_end = args.beta_end
    beta_steps = args.beta_steps
    path = args.path
    tag = args.tag

    kwargs["initial_h"] = func_from_str(args.initial_h[0])
    kwargs["initial_u"] = func_from_str(args.initial_u[0])

    # Setup time-step
    dt_func = lambda beta: (lambda dx, U, H, g: beta * dx / np.sqrt(g * H))

    # Set step sizes
    beta_values = np.linspace(beta_start, beta_end, beta_steps)

    # Compute the errors
    errors = np.zeros((len(beta_values), 2))
    for i, beta in enumerate(beta_values):
        params = Parameters.from_dict(**kwargs, dt=dt_func(beta))

        num_solver = ForwardBackwardSolver(params)
        exact_solver = AnalyticSolver(params)

        n_steps = int(final_time / params.dt)
        for _ in range(n_steps):
            num_solver.step()
            exact_solver.step()

        h_error = np.linalg.norm(num_solver.h - exact_solver.h, np.inf)
        u_error = np.linalg.norm(num_solver.u - exact_solver.u, np.inf)
        print(
            f"beta = {beta_values[i]}, h_error = {h_error}, "
            f"u_error = {u_error}, final_time = {n_steps * params.dt}"
        )

        errors[i] = [h_error, u_error]

    # Plot the errors
    plt.figure(figsize=(8, 6))
    plt.plot(beta_values, errors[:, 0], "k-", label=r"$h$")
    plt.plot(beta_values, errors[:, 1], "k--", label=r"$\bar{u}$")

    plt.ylim([np.min(errors) / 2, 1])
    plt.yscale("log")

    plt.xlabel(r"$\beta$")
    plt.ylabel("Error")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"{path}stability_sweep{tag}.png", dpi=300)
