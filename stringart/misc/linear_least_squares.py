import os
from pathlib import Path

import numpy as np
import scipy
from matplotlib import pyplot as plt
from stringart.line_algorithms.matrix import MatrixGenerator
from stringart.misc.least_squares_rounding_trial import compute_solution
from stringart.utils.image import ImageWrapper
from stringart.utils.performance_analysis import Benchmark
from stringart.utils.types import Method, Mode

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
image_mode = "center"
number_of_pegs = 100
method = "sparse"


def linear_programming(
    shape: tuple[int, ...],
    number_of_pegs: int = 100,
    image_mode: Mode = "center",
    method: Method = "sparse",
) -> tuple[np.ndarray, np.ndarray]:
    A, _ = MatrixGenerator.compute_matrix(shape, number_of_pegs, image_mode, method)
    b: np.ndarray = ImageWrapper.flatten_image(image)

    optimize_results = scipy.optimize.lsq_linear(A, b, bounds=(0, np.inf))

    return A, optimize_results.x


def main():
    stringart_directory: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory: Path = stringart_directory.parent.resolve()

    Benchmark.initialize_metadata(directory)
    benchmark = Benchmark(image, image_mode, number_of_pegs)
    benchmarks_result = benchmark.run_benchmark(
        linear_programming,
        shape=shape,
        number_of_pegs=number_of_pegs,
        image_mode=image_mode,
        method=method,
    )

    benchmark.save_benchmarks([benchmarks_result], "linear_least_squares")
    A, x = linear_programming(shape, number_of_pegs, image_mode, method)
    solution = compute_solution(A, x)

    plt.imshow(solution, cmap="grey")
    plt.show()
    plt.imsave("../../outputs/misc/linear_least_squares_lena.png", solution, cmap="grey")


if __name__ == "__main__":
    main()
