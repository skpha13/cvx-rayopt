import json
import logging
import os
from pathlib import Path

from matplotlib import pyplot as plt
from skimage import io

from stringart.solver import Solver
from stringart.utils.image import ImageWrapper
from stringart.utils.performance_analysis import BenchmarkResult, benchmark, run_benchmarks, save_benchmarks
from stringart.utils.types import Metadata, Mode


def main() -> None:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    stringart_directory: Path = Path(os.path.dirname(os.path.abspath(__file__)))
    directory: Path = stringart_directory.parent.resolve()

    metadata = Metadata(directory)

    image = ImageWrapper()
    image.read_bw(metadata.path / "imgs/lena.png")
    mode: Mode = "center"

    # solver = Solver(image, mode, number_of_pegs=100)
    # params = {"method": "sparse"}
    # benchmark_result: BenchmarkResult = benchmark(solver.least_squares, **params)
    # solution = benchmark_result.output_image
    #
    # logger.info(str(benchmark_result))

    # solver = Solver(image, mode, number_of_pegs=50)
    # solution = solver.greedy(number_of_lines=1000, selector_type="random")

    # io.imsave(metadata.path / "outputs/lena_stringart_greedy.png", solution)
    # io.imshow(solution)
    # plt.show()

    results = run_benchmarks(image)

    formatted_results = "\n\n".join([str(result) for result in results])
    logger.info(formatted_results)

    dir_path = metadata.path / "benchmarks"
    file_path = dir_path / "benchmarks_01.json"
    os.makedirs(dir_path, exist_ok=True)
    save_benchmarks(results, file_path)


if __name__ == "__main__":
    main()
