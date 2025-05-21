import os
from pathlib import Path

from matplotlib import pyplot as plt
from stringart.solver import Solver
from stringart.utils.image import ImageWrapper, crop_image
from stringart.utils.perf_analyzer import Benchmark
from stringart.utils.types import CropMode, MatrixRepresentation, Rasterization

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
number_of_pegs = 100
matrix_representation: MatrixRepresentation = "sparse"
rasterization: Rasterization = "xiaolin-wu"
lambd = 100
regularizer = "smooth"


def solve():
    stringart_directory: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory: Path = stringart_directory.parent.resolve()

    image_cropped = crop_image(image, crop_mode)
    solver = Solver(image_cropped, crop_mode, number_of_pegs=number_of_pegs, rasterization=rasterization)

    print("Regularizer: None")
    Benchmark.initialize_metadata(directory)
    benchmark = Benchmark(image, crop_mode, number_of_pegs, rasterization)
    benchmarks_reg_none = benchmark.run_benchmark(
        solver.ls_regularized,
        matrix_representation=matrix_representation,
        lambd=lambd,
        regularizer=None,
    )

    print("Regularizer: smooth")
    benchmarks_reg_smooth = benchmark.run_benchmark(
        solver.ls_regularized,
        matrix_representation=matrix_representation,
        lambd=lambd,
        regularizer="smooth",
    )

    print("Regularizer: abs")
    benchmarks_reg_abs = benchmark.run_benchmark(
        solver.ls_regularized,
        matrix_representation=matrix_representation,
        lambd=lambd,
        regularizer="abs",
    )

    benchmark.save_benchmarks(
        [benchmarks_reg_none, benchmarks_reg_smooth, benchmarks_reg_abs], "least_squares_regularized"
    )


if __name__ == "__main__":
    solve()
