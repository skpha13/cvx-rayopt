import argparse
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import get_args

from stringart.cli_functions import Configuration
from stringart.mp.greedy_selector import GreedySelector
from stringart.utils.types import (
    CropMode,
    MatchingPursuitMethod,
    MatrixRepresentation,
    Metadata,
    QPSolvers,
    Rasterization,
    RegularizationType,
)

SOLVE_COMMAND_NAME = "solve"


def add_arguments(parser: ArgumentParser) -> ArgumentParser:
    """Add arguments to the parser for different commands and options."""
    parser.add_argument(
        "--log-level",
        choices=["INFO", "WARNING"],
        default="INFO",
        required=False,
        help="Set the logging level (default: INFO).",
    )

    subparsers = parser.add_subparsers(title="Commands", dest="command")
    # subcommand: run_benchmarks
    benchmarks_parser = subparsers.add_parser("run-benchmarks", help="Run all benchmarks for StringArt.")
    benchmarks_parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Name of the directory where benchmark results will be saved. Default: benchmarks_01",
    )
    # subcommand: run_analysis
    analysis_parser = subparsers.add_parser("run-analysis", help="Run analysis on StringArt benchmarks.")
    analysis_parser.add_argument(
        "--input-benchmark-dir",
        type=str,
        required=True,
        help="Directory name for the benchmark output to load benchmarks.",
    )
    analysis_parser.add_argument(
        "--analysis-name",
        type=str,
        required=False,
        help="Name of the directory containing the benchmark results to analyze. Default: analysis_01",
    )

    # subcommand: solve
    solve_parser = subparsers.add_parser(SOLVE_COMMAND_NAME, help="Compute StringArt configurations.")
    solver_subparsers = solve_parser.add_subparsers(title="Solvers", dest="solver", required=True)

    # Least Squares Solver
    ls_parser = solver_subparsers.add_parser("ls", help="Least Squares solver options.")
    ls_parser.add_argument(
        "--matrix-representation",
        choices=get_args(MatrixRepresentation),
        required=False,
        help="Matrix representation method. Defaults to `sparse`.",
    )
    ls_parser.add_argument(
        "--number-of-lines",
        type=int,
        required=False,
        help="Top K number of lines to select.",
    )
    ls_parser.add_argument(
        "--binary",
        type=bool,
        required=False,
        help="If set, projects the solution vector `x` to binary values (0 or 1).",
    )

    # Linear Least Squares Solver
    lls_parser = solver_subparsers.add_parser("lls", help="Linear Least Squares solver options.")
    lls_parser.add_argument(
        "--matrix-representation",
        choices=get_args(MatrixRepresentation),
        required=False,
        help="Matrix representation method. Defaults to `sparse`.",
    )
    lls_parser.add_argument(
        "--number-of-lines",
        type=int,
        required=False,
        help="Top K number of lines to select.",
    )
    lls_parser.add_argument(
        "--binary",
        type=bool,
        required=False,
        help="If set, projects the solution vector `x` to binary values (0 or 1).",
    )

    # Matching Pursuit Solver
    mp_parser = solver_subparsers.add_parser("mp", help="Matching Pursuit solver options.")
    mp_parser.add_argument(
        "--method",
        choices=get_args(MatchingPursuitMethod),
        required=False,
        help="Algorithm selection, either Greedy or Orthogonal Matching Pursuit. Defaults to `orthogonal`.",
    )
    mp_parser.add_argument(
        "--selector",
        choices=get_args(GreedySelector),
        required=False,
        help="Selector method to use (only applicable to matching-pursuit with greedy method). Defaults to `dot-product`.",
    )
    mp_parser.add_argument(
        "--number-of-lines",
        type=int,
        required=True,
        help="Top K number of lines to select.",
    )

    # Binary Projection Solver
    bpls_parser = solver_subparsers.add_parser("bpls", help="Binary Projection Least Squares options.")
    bpls_parser.add_argument(
        "--qp-solver",
        choices=get_args(QPSolvers),
        required=False,
        help="Quadratic programming solver to use for the least squares step. Defaults to 'cvxopt'.",
    )
    bpls_parser.add_argument(
        "--matrix-representation",
        choices=get_args(MatrixRepresentation),
        required=False,
        help="Matrix representation method. Defaults to `sparse`.",
    )
    bpls_parser.add_argument(
        "--k",
        type=int,
        required=False,
        help="Number of variables to fix to 1 in each iteration. Controls the granularity of the binary projection step.",
    )
    bpls_parser.add_argument(
        "--max-iterations",
        type=int,
        required=False,
        help="Maximum number of iterations to run the binary projection solver before stopping.",
    )
    bpls_parser.add_argument(
        "--lambda",
        type=float,
        required=False,
        help="The regularization strength. Defaults to None",
    )

    ls_reg = solver_subparsers.add_parser("lsr", help="Least Squares Regularized options.")
    ls_reg.add_argument(
        "--matrix-representation",
        choices=get_args(MatrixRepresentation),
        required=False,
        help="Matrix representation method. Defaults to `sparse`.",
    )
    ls_reg.add_argument(
        "--regularizer",
        choices=get_args(RegularizationType),
        required=False,
        help="Regularization method. Defaults to None.",
    )
    ls_reg.add_argument(
        "--lambda",
        type=float,
        required=False,
        help="The regularization strength. Defaults to 0.1",
    )

    # Common Arguments
    for subparser in [ls_parser, lls_parser, mp_parser, bpls_parser, benchmarks_parser, analysis_parser, ls_reg]:
        subparser.add_argument(
            "--image-path",
            type=str,
            required=True,
            help="Path to the input image. Supported formats: PNG, JPEG, etc.",
        )
        subparser.add_argument(
            "--number-of-pegs",
            type=int,
            required=False,
            help="Number of pegs to use. Defaults to 100.",
        )
        subparser.add_argument(
            "--crop-mode",
            choices=get_args(CropMode),
            required=False,
            help="Crop mode to use. Defaults to `center`.",
        )
        subparser.add_argument(
            "--rasterization",
            choices=get_args(Rasterization),
            required=False,
            help="Line rasterization algorithm. Use 'bresenham' for fast lines, or 'xiaolin-wu' for anti-aliased lines.",
        )
        subparser.add_argument(
            "--block-size",
            type=int,
            required=False,
            help="Enables residual computation using supersampling. Example values: 2, 4, 8, 16. Defaults to `None`.",
        )

    return parser


def validate_arguments(args):
    """Validate arguments for logical consistency."""

    if args.command == SOLVE_COMMAND_NAME and args.solver == "mp":
        if args.number_of_lines is None:
            raise ValueError("The `number-of-lines` argument can not be None with the matching-pursuit solver.")

        if args.block_size is None:
            raise ValueError("The `block-size` argument can not be None with the matching-pursuit solver.")


def main() -> None:
    stringart_directory: Path = Path(os.path.dirname(os.path.abspath(__file__)))
    directory: Path = stringart_directory.parent.resolve()
    metadata = Metadata(directory)

    parser = argparse.ArgumentParser(prog="StringArt", description="StringArt CLI")
    add_arguments(parser)
    args = parser.parse_args()
    validate_arguments(args)

    logging.basicConfig(level=getattr(logging, args.log_level))

    configuration = Configuration(
        metadata=metadata,
        command=args.command,
        solver=getattr(args, "solver", None),
        image_path=getattr(args, "image_path", None),
        number_of_pegs=getattr(args, "number_of_pegs", None),
        crop_mode=getattr(args, "crop_mode", None),
        rasterization=getattr(args, "rasterization", "xiaolin-wu"),
        matrix_representation=getattr(args, "matrix_representation", None),
        mp_method=getattr(args, "method", None),
        number_of_lines=getattr(args, "number_of_lines", None),
        selector_type=getattr(args, "selector", None),
        binary=getattr(args, "binary", None),
        block_size=getattr(args, "block_size", None),
        qp_solver=getattr(args, "qp_solver", None),
        k=getattr(args, "k", None),
        max_iterations=getattr(args, "max_iterations", None),
        regularizer=getattr(args, "regularizer", None),
        lambd=getattr(args, "lambda", None),
        input_benchmark_dir=getattr(args, "input_benchmark_dir", None),
        output_dir=getattr(args, "output_dir", None),
    )

    configuration.run_configuration()


if __name__ == "__main__":
    main()
