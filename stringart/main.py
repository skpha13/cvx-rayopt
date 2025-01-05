import logging
import os
from pathlib import Path

from matplotlib import pyplot as plt
from skimage import io

from stringart.solver import Solver
from stringart.utils.analysis import BenchmarkResult, benchmark
from stringart.utils.image import ImageWrapper
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

    solver = Solver(image, mode, number_of_pegs=100)
    benchmark_result: BenchmarkResult = benchmark(solver.least_squares, "sparse")
    solution = benchmark_result.output_image

    # solver = Solver(image, mode, number_of_pegs=50)
    # solution = solver.greedy(number_of_lines=1000, selector_type="random")

    logger.info(benchmark_result)

    io.imsave(metadata.path / "outputs/lena_stringart_greedy.png", solution)
    io.imshow(solution)
    plt.show()


if __name__ == "__main__":
    main()
