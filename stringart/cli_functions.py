import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from skimage import io

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
        solver = Solver(image, self.crop_mode, number_of_pegs=self.number_of_pegs)
        benchmark = Benchmark(image=image, mode=self.crop_mode, number_of_pegs=self.number_of_pegs)

        if self.command == "solve":
            solution: np.ndarray | None = None
            save_path = self.metadata.path / "outputs/lena_stringart_sparse.png"

            if self.solver == "least-squares":
                solution = solver.least_squares(self.matrix_representation)
            else:
                solution = solver.matching_pursuit(self.number_of_lines, self.method, self.selector)

            plt.axis("off")
            plt.title("Computed Image")
            io.imshow(solution)
            io.imsave(save_path, solution)
            plt.show()

            logger.info(f"Image saved to: {save_path}")
        elif self.command == "run-benchmarks":
            results = benchmark.run_benchmarks()

            formatted_results = "\n\n".join([str(result) for result in results])
            logger.info(formatted_results)

            # TODO: user should be able to change the filename
            Benchmark.save_benchmarks(results, "benchmarks_01")

            logger.info(f"Benchmarks saved to: {Benchmark.BENCHMARKS_PATH}")
        elif self.command == "run-analysis":
            # TODO: give name of benchmark output
            benchmarks = Benchmark.load_benchmarks("benchmarks_01")

            # TODO: configurable name for analysis
            benchmark.run_analysis(
                benchmarks=benchmarks,
                ground_truth_image=image,
                dirname="analysis_01",
            )

            logger.info(f"Analysis saved to: {Benchmark.PLOTS_PATH}")
