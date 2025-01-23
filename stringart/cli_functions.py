import logging
import os.path
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from stringart.solver import Solver
from stringart.utils.image import ImageWrapper
from stringart.utils.performance_analysis import Benchmark
from stringart.utils.types import Metadata
from utils.greedy_selector import GreedySelector
from utils.types import Method, Mode

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Configuration:
    metadata: Metadata
    command: str
    solver: str
    image_path: str | Path
    number_of_pegs: int | None
    crop_mode: Mode | None
    matrix_representation: str
    method: Method | None
    number_of_lines: int
    selector: GreedySelector | None

    def run_configuration(self):
        image = ImageWrapper.read_bw(self.image_path)
        image_name = os.path.splitext(os.path.basename(self.image_path))[0]

        if self.command == "solve":
            solver = Solver(image, self.crop_mode, number_of_pegs=self.number_of_pegs)
            solution: np.ndarray | None = None
            save_path = self.metadata.path / "outputs" / f"{image_name}.png"

            if self.solver == "least-squares":
                solution = solver.least_squares(self.matrix_representation)
            else:
                solution = solver.matching_pursuit(self.number_of_lines, self.method, selector=self.selector)

            plt.axis("off")
            plt.title("Computed Image")
            plt.imshow(solution, cmap="grey")
            plt.imsave(save_path, solution, cmap="grey")
            plt.show()

            logger.info(f"Image saved to: {save_path}")
            return

        Benchmark.initialize_metadata(self.metadata.path)
        benchmark = Benchmark(image=image, mode=self.crop_mode, number_of_pegs=self.number_of_pegs)

        if self.command == "run-benchmarks":
            results = benchmark.run_benchmarks()

            formatted_results = "\n\n".join([str(result) for result in results])
            logger.info(formatted_results)

            # TODO: user should be able to change the filename
            Benchmark.save_benchmarks(results, "benchmarks_01")

            logger.info(f"Benchmarks saved to: {Benchmark.BENCHMARKS_PATH}")
            return

        if self.command == "run-analysis":
            # TODO: give name of benchmark output
            benchmarks = Benchmark.load_benchmarks("benchmarks_01")

            # TODO: configurable name for analysis
            benchmark.run_analysis(
                benchmarks=benchmarks,
                ground_truth_image=image,
                dirname="analysis_01",
            )

            logger.info(f"Analysis saved to: {Benchmark.PLOTS_PATH}")
