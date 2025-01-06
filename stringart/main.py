import json
import logging
import os
from pathlib import Path
from typing import List

from matplotlib import pyplot as plt
from skimage import io

from stringart.solver import Solver
from stringart.utils.image import ImageWrapper
from stringart.utils.performance_analysis import Benchmark, BenchmarkResult
from stringart.utils.types import Metadata, Mode


def main() -> None:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    stringart_directory: Path = Path(os.path.dirname(os.path.abspath(__file__)))
    directory: Path = stringart_directory.parent.resolve()

    metadata = Metadata(directory)

    image = ImageWrapper.read_bw(metadata.path / "imgs/lena.png")
    ground_truth_image = image.copy()
    mode: Mode = "center"
    Benchmark.initialize_metadata(metadata.path)

    # solver = Solver(image, mode, number_of_pegs=100)
    # params = {"method": "sparse"}
    # benchmark = Benchmark(image=image, mode=mode, path=metadata.path, number_of_pegs=100)
    # benchmark_result: BenchmarkResult = benchmark.run_benchmark(solver.least_squares, **params)
    # benchmark.save_benchmarks([benchmark_result])
    # solution = benchmark_result.output_image
    #
    # logger.info(str(benchmark_result))
    #
    # io.imsave(metadata.path / "outputs/lena_stringart_sparse.png", solution)
    # io.imshow(solution)
    # plt.show()

    benchmark = Benchmark(image=image, mode=mode, number_of_pegs=100)
    results = benchmark.run_benchmarks()

    formatted_results = "\n\n".join([str(result) for result in results])
    logger.info(formatted_results)

    Benchmark.save_benchmarks(results, "benchmarks_01")

    benchmarks = Benchmark.load_benchmarks("benchmarks_01")
    benchmark.run_analysis(
        benchmarks=benchmarks,
        ground_truth_image=ground_truth_image,
        dirname="analysis_01",
    )


if __name__ == "__main__":
    main()
