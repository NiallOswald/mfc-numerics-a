"""Utilities for the scripts."""

from argparse import ArgumentParser
from math import *  # noqa: F401, F403  # This is unsafe but is needed for the eval


def func_from_str(func_str: str, vars: str = "x"):
    """Return a function from a string."""
    return eval(f"lambda {vars}: {func_str}")


def parameter_parser(
    parser: ArgumentParser, allow_U: bool = True, allow_dt: bool = True  # noqa: N803
):
    """Add the parameters for the Saint-Venant equation to the parser."""
    parser.add_argument(
        "--start_point",
        type=float,
        default=0.0,
        help="The start point of the domain.",
    )
    parser.add_argument(
        "--end_point", type=float, default=1.0, help="The end point of the domain."
    )
    parser.add_argument(
        "initial_h",
        type=str,
        nargs=1,
        help="An expression in the coordinate x for the inital condition of "
        "the variable h. The function should be a quoted string. E.g. "
        '"sin(x)". The function must be periodic and always return a float.',
    )
    parser.add_argument(
        "initial_u",
        type=str,
        nargs=1,
        help="An expression in the coordinate x for the inital condition of "
        "the variable u. The function should be a quoted string. E.g. "
        '"sin(x)". The function must be periodic and always return a float.',
    )
    parser.add_argument(
        "--g", type=float, default=9.81, help="The acceleration due to gravity."
    )
    parser.add_argument(
        "--theta", type=float, default=0.0, help="The angle of the bed."
    )
    parser.add_argument(
        "--H", type=float, default=1.0, help="The reference depth of the fluid."
    )

    if allow_U:
        parser.add_argument(
            "--U", type=float, default=1.0, help="The reference velocity of the fluid."
        )

    if allow_dt:
        parser.add_argument(
            "--dt",
            type=str,
            default="dx",
            help="An expression in the parameters dx, U, H, g for the time-step dt. "
            'The function should be a quoted string. E.g. "dx**2". The function must be '
            "periodic and always return a float.",
        )


def path_parser(parser: ArgumentParser):
    """Add the path argument to the parser."""
    parser.add_argument(
        "--path", type=str, default="./", help="The path to save the figure to."
    )
    parser.add_argument(
        "--tag", type=str, default="", help="A tag to add to the saved figure."
    )
