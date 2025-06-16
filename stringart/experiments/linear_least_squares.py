import os
from pathlib import Path

import numpy as np
import scipy
from matplotlib import pyplot as plt
from stringart.experiments.least_squares_rounding import compute_solution
from stringart.line.matrix import MatrixGenerator
from stringart.utils.image import ImageWrapper
from stringart.utils.perf_analyzer import Benchmark
from stringart.utils.types import CropMode, MatrixRepresentation, Rasterization

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
number_of_pegs = 100
matrix_representation: MatrixRepresentation = "sparse"


def linear_least_squares(
    src: np.ndarray,
    shape: tuple[int, ...],
    number_of_pegs: int = 100,
    crop_mode: CropMode = "center",
    matrix_representation: MatrixRepresentation = "sparse",
    rasterization: Rasterization = "bresenham",
) -> tuple[np.ndarray, np.ndarray, list[np.floating]]:
    A = MatrixGenerator.compute_matrix(shape, number_of_pegs, crop_mode, matrix_representation, rasterization)
    b: np.ndarray = ImageWrapper.flatten_image(src)

    optimize_results = scipy.optimize.lsq_linear(A, b, bounds=(0, np.inf))

    x = optimize_results.x
    residual = np.linalg.norm(b - A @ x)

    return A, optimize_results.x, [residual]


def main():
    stringart_directory: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory: Path = stringart_directory.parent.resolve()

    Benchmark.initialize_metadata(directory)
    benchmark = Benchmark(image, crop_mode, number_of_pegs)
    benchmarks_result = benchmark.run_benchmark(
        linear_least_squares,
        src=image,
        shape=shape,
        number_of_pegs=number_of_pegs,
        crop_mode=crop_mode,
        matrix_representation=matrix_representation,
    )

    benchmark.save_benchmarks([benchmarks_result], "linear_least_squares")
    A, x, _ = linear_least_squares(image, shape, number_of_pegs, crop_mode, matrix_representation)
    solution = compute_solution(A, x)

    plt.imshow(solution, cmap="gray")
    plt.show()
    plt.imsave("../../outputs/misc/linear_least_squares_lena.png", solution, cmap="gray")


if __name__ == "__main__":
    main()
