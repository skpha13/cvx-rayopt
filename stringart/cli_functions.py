import logging
import os.path
import time
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import numpy as np
from matplotlib import pyplot as plt

from stringart.mp.greedy_selector import GreedySelector
from stringart.solver import Solver
from stringart.utils.image import ImageWrapper
from stringart.utils.perf_analyzer import Benchmark
from stringart.utils.time_memory_format import convert_monotonic_time, format_time
from stringart.utils.types import (
    CropMode,
    ElapsedTime,
    MatchingPursuitMethod,
    MatrixRepresentation,
    Metadata,
    QPSolvers,
    Rasterization,
    RegularizationType,
    SolverType,
)

logger = logging.getLogger(__name__)


@dataclass
class Configuration:
    metadata: Metadata
    command: str
    solver: SolverType
    image_path: str | Path
    number_of_pegs: int | None = None
    crop_mode: CropMode | None = None
    rasterization: Rasterization | None = None
    matrix_representation: MatrixRepresentation | None = None
    mp_method: MatchingPursuitMethod | None = None
    number_of_lines: int | None = None
    selector_type: GreedySelector | None = None
    binary: bool | None = None
    qp_solver: QPSolvers | None = None
    k: int | None = None
    max_iterations: int | None = None
    regularizer: RegularizationType | None = None
    lambd: float | None = None
    block_size: int | None = None

    output_dir: str | None = None
    input_benchmark_dir: str | None = None

    def _get_solver_instance(self) -> Solver:
        image = ImageWrapper.read_bw(self.image_path)
        return Solver(
            image,
            self.crop_mode,
            number_of_pegs=self.number_of_pegs,
            rasterization=self.rasterization,
            block_size=self.block_size,
        )

    def _solve(self, solver: Solver) -> np.ndarray:
        # fmt: off
        solver_methods = {
            "ls": lambda: solver.ls(self.matrix_representation, self.number_of_lines, self.binary),
            "lls": lambda: solver.lls(self.matrix_representation, self.number_of_lines, self.binary),
            "mp": lambda: solver.mp(self.number_of_lines, self.mp_method, selector_type=self.selector_type),
            "bpls": lambda: solver.bpls(self.qp_solver, self.matrix_representation, self.k, self.max_iterations),
            "lsr": lambda: solver.lsr(self.matrix_representation, self.regularizer, self.lambd),
            "radon": lambda: solver.radon(),
        }
        # fmt: on

        if self.solver not in solver_methods:
            raise ValueError(f"Unsupported solver type: {self.solver}. Supported solvers are: {get_args(SolverType)}")

        A, x, _ = solver_methods[self.solver]()
        return solver.compute_solution(A, x, uds=self.block_size is not None)

    def run_config_lite(self) -> np.ndarray:
        if self.command != "solve":
            raise RuntimeError(f"`run_config_lite` only supports the 'solve' command, but got '{self.command}'.")

        solver = self._get_solver_instance()
        return self._solve(solver)

    def run_configuration(self, running_tests: bool = False):
        image = ImageWrapper.read_bw(self.image_path)
        image_name = os.path.splitext(os.path.basename(self.image_path))[0]

        if self.command == "solve":
            solver = self._get_solver_instance()
            save_path = self.metadata.path / "outputs" / f"{image_name}.png"

            time_start = time.monotonic()
            solution = self._solve(solver)
            time_end = time.monotonic()
            elapsed_monotonic_time = time_end - time_start
            elapsed_time: ElapsedTime = convert_monotonic_time(elapsed_monotonic_time)
            logger.info(f"Elapsed Time: {format_time(elapsed_time)}")

            if not running_tests:
                plt.axis("off")
                plt.title("Computed Image")
                plt.imshow(solution, cmap="gray")
                plt.imsave(save_path, solution, cmap="gray")
                plt.show()

                logger.info(f"Image saved to: {save_path}")

            return

        Benchmark.initialize_metadata(self.metadata.path)
        benchmark = Benchmark(
            image=image,
            crop_mode=self.crop_mode,
            number_of_pegs=self.number_of_pegs,
            rasterization=self.rasterization,
            block_size=self.block_size,
        )

        if self.command == "run-benchmarks":
            output_dir = self.output_dir if self.output_dir else "benchmarks_01"
            results = benchmark.run_benchmarks()

            formatted_results = "\n\n".join([str(result) for result in results])
            logger.info(formatted_results)

            if not running_tests:
                Benchmark.save_benchmarks(results, output_dir)

            return

        if self.command == "run-analysis":
            input_dir = self.input_benchmark_dir
            output_dir = self.output_dir if self.output_dir else "analysis_01"

            benchmarks = Benchmark.load_benchmarks(input_dir)

            benchmark.run_analysis(
                benchmarks=benchmarks,
                ground_truth_image=1 - image,
                dirname=output_dir,
            )
