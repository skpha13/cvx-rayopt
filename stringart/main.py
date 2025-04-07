import argparse
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import get_args

from stringart.cli_functions import Configuration
from stringart.utils.greedy_selector import GreedySelector
from stringart.utils.types import (
    CropMode,
    MatchingPursuitMethod,
    MatrixRepresentation,
    Metadata,
    Rasterization,
    SolverType,
)

SOLVE_COMMAND_NAME = "solve"


def add_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Add arguments to the parser for different commands and options."""
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    # subcommand: run_benchmarks
    subparsers.add_parser("run-benchmarks", help="Run all benchmarks for StringArt.")
    # subcommand: run_analysis
    subparsers.add_parser("run-analysis", help="Run analysis on StringArt benchmarks.")
    # subcommand: compute
    compute_parser = subparsers.add_parser(SOLVE_COMMAND_NAME, help="Compute StringArt configurations.")

    compute_parser.add_argument(
        "--solver",
        choices=get_args(SolverType),
        required=True,
        help="Solver to use for computation.",
    )

    # Least Squares Group
    least_squares_group = compute_parser.add_argument_group(
        "least-squares arguments", "Options specific to the least-squares solver."
    )
    least_squares_group.add_argument(
        "--matrix-representation",
        choices=get_args(MatrixRepresentation),
        required=False,
        help="Matrix representation method. Defaults to `sparse`.",
    )

    # Matching Pursuit Group
    matching_pursuit_group = compute_parser.add_argument_group(
        "matching-pursuit arguments", "Options specific to the matching-pursuit solver."
    )
    matching_pursuit_group.add_argument(
        "--method",
        choices=get_args(MatchingPursuitMethod),
        required=False,
        help="Algorithm selection, either Greedy or Orthogonal Matching Pursuit. Defaults to `orthogonal`.",
    )
    matching_pursuit_group.add_argument(
        "--selector",
        choices=get_args(GreedySelector),
        required=False,
        help="Selector method to use (only applicable to matching-pursuit with greedy method). Defaults to `dot-product`.",
    )

    # Common Arguments
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="The file path to the image you want to process. The path can be absolute or relative, and the image should be in a supported format (e.g., PNG, JPEG).",
    )
    parser.add_argument(
        "--number-of-pegs",
        type=int,
        required=False,
        help="Number of pegs to use in the computation. Defaults to 100.",
    )
    parser.add_argument(
        "--crop-mode",
        choices=get_args(CropMode),
        required=False,
        help="Crop mode to use on the provided image. Default to `center`.",
    )
    parser.add_argument(
        "--rasterization",
        choices=get_args(Rasterization),
        required=False,
        help="Specifies the line rasterization algorithm to use. "
        "'bresenham' is an efficient integer-based algorithm for drawing lines, "
        "while 'xiaolin-wu' produces anti-aliased lines for smoother results.",
    )
    parser.add_argument(
        "--number-of-lines",
        type=int,
        required=False,
        help="Top K number of lines to select.",
    )

    return parser


def validate_arguments(args):
    """Validate arguments for logical consistency."""

    if args.command == SOLVE_COMMAND_NAME and args.solver == "matching-pursuit":
        if args.number_of_lines is None:
            raise ValueError("The `number-of-lines` argument can not be None with the matching-pursuit solver.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    stringart_directory: Path = Path(os.path.dirname(os.path.abspath(__file__)))
    directory: Path = stringart_directory.parent.resolve()
    metadata = Metadata(directory)

    parser = argparse.ArgumentParser(prog="StringArt", description="StringArt CLI")
    add_arguments(parser)
    args = parser.parse_args()
    validate_arguments(args)

    configuration = Configuration(
        metadata=metadata,
        command=args.command,
        solver=getattr(args, "solver", None),
        image_path=getattr(args, "image_path", None),
        number_of_pegs=getattr(args, "number_of_pegs", None),
        crop_mode=getattr(args, "crop_mode", None),
        rasterization=getattr(args, "rasterization", "bresenham"),
        matrix_representation=getattr(args, "matrix_representation", None),
        mp_method=getattr(args, "method", None),
        number_of_lines=getattr(args, "number_of_lines", None),
        selector_type=getattr(args, "selector", None),
    )

    configuration.run_configuration()


if __name__ == "__main__":
    main()
