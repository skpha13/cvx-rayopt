import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from stringart.line_algorithms.matrix import MatrixGenerator
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


def main():
    stringart_directory: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory: Path = stringart_directory.parent.resolve()

    image_cropped = crop_image(image, crop_mode)
    solver = Solver(image_cropped, crop_mode, number_of_pegs=number_of_pegs, rasterization=rasterization)

    Benchmark.initialize_metadata(directory)
    benchmark = Benchmark(image, crop_mode, number_of_pegs, rasterization)
    result = benchmark.run_benchmark(
        solver.least_squares,
        matrix_representation=matrix_representation,
    )
    gt = np.array(result.output_image)

    benchmark_results = [result]
    benchmark.save_benchmarks(benchmark_results, "benchmark_experiment")

    result = benchmark.load_benchmarks("benchmark_experiment")[0]
    print(result)

    A, _ = MatrixGenerator.compute_matrix(
        result.shape,
        result.number_of_pegs,
        result.crop_mode,
        matrix_representation=result.params["matrix_representation"],
        rasterization=result.rasterization,
    )
    x = np.array(result.x)

    reconstructed_image = solver.compute_solution(A, x)
    b = solver.b

    residual = np.linalg.norm(b - A @ x)
    print(f"\nResidual Ground Truth: {result.residual_history[-1]}\nResidual: {residual}")

    plt.imshow(gt, cmap="grey")
    plt.axis("off")
    plt.show()

    plt.imshow(reconstructed_image, cmap="grey")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
