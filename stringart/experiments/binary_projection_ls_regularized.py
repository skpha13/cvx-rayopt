import logging
import os
from pathlib import Path

from stringart.solver import Solver
from stringart.utils.image import ImageWrapper, crop_image
from stringart.utils.perf_analyzer import Benchmark
from stringart.utils.types import CropMode, MatrixRepresentation, Rasterization

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
number_of_pegs = 50
matrix_representation: MatrixRepresentation = "sparse"
rasterization: Rasterization = "xiaolin-wu"
k = 10
max_iterations = 100
lambd = 0.1


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    stringart_directory: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory: Path = stringart_directory.parent.resolve()

    image_cropped = crop_image(image, crop_mode)
    solver = Solver(image_cropped, crop_mode, number_of_pegs=number_of_pegs, rasterization=rasterization)

    Benchmark.initialize_metadata(directory)
    benchmark = Benchmark(image, crop_mode, number_of_pegs, rasterization)
    cvxopt_result = benchmark.run_benchmark(
        solver.binary_projection_ls,
        solver="cvxopt",
        matrix_representation=matrix_representation,
        k=k,
        max_iterations=max_iterations,
        lambd=lambd,
    )

    benchmark_results = [cvxopt_result]
    benchmark.save_benchmarks(benchmark_results, "binary_projection_regularized")
    benchmark.run_analysis(benchmark_results, image, "binary_projection_regularized")


if __name__ == "__main__":
    main()
