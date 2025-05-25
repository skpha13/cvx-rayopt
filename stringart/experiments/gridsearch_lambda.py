import logging
import os
from pathlib import Path

import numpy as np
from stringart.experiments.least_squares_regularized import bin_and_plot
from stringart.line.matrix import MatrixGenerator
from stringart.optimize.regularization import WeightedRegularizer
from stringart.solver import Solver
from stringart.utils.image import ImageWrapper, crop_image
from stringart.utils.perf_analyzer import Benchmark, BenchmarkResult
from stringart.utils.types import CropMode, MatrixRepresentation, Rasterization
from tqdm import tqdm

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
number_of_pegs = 50
matrix_representation: MatrixRepresentation = "sparse"
rasterization: Rasterization = "xiaolin-wu"
k = 3
max_iterations = 1000
lambds = [0.1, 1, 10, 100, 1000, 10_000]


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_benchmarks(benchmark: Benchmark, solver: Solver):
    results: list[BenchmarkResult] = []
    for lambd in lambds:
        result = benchmark.run_benchmark(
            solver.bpls,
            solver="cvxopt",
            matrix_representation=matrix_representation,
            k=k,
            max_iterations=max_iterations,
            lambd=lambd,
        )

        results.append(result)

    benchmark.save_benchmarks(results, "gridsearch_lambda")
    benchmark.run_analysis(results, image, "gridsearch_lambda")


def solve_two_iterations(solver: Solver, lambd: float) -> np.ndarray | None:
    A, _ = MatrixGenerator.compute_matrix(shape, number_of_pegs, crop_mode, matrix_representation, rasterization)

    n = A.shape[1]
    x_fixed = np.full(n, np.nan)
    set1 = set()

    regularizer = WeightedRegularizer(n)
    x_free = None
    for _ in range(2):
        free_indices = np.isnan(x_fixed)

        A_free = A[:, free_indices]
        b_adjusted = solver.b.copy()

        if set1:
            A_fixed_1 = A[:, list(set1)]
            b_adjusted -= A_fixed_1 @ np.ones(len(set1))

        x_free = solver.solve_qp_cvxopt(A_free, b_adjusted, regularizer, lambd)

        free_idx_array = np.where(free_indices)[0]
        top_k = min(k, len(x_free))
        top_k_indices = np.argsort(-x_free)[:top_k]
        chosen_indices = free_idx_array[top_k_indices]

        for idx in chosen_indices:
            x_fixed[idx] = 1
            set1.add(idx)

        if np.all(~np.isnan(x_fixed)):
            break

        x_free_weights = np.delete(x_free, top_k_indices)
        regularizer.update_weights(x_free_weights)

    return x_free


def solve_plot_x(solver: Solver):
    xs = []
    labels = [str(l) for l in lambds]
    for lambd in tqdm(lambds, desc="Gridsearch lambda"):
        logger.info(f"Lambda {lambd}")

        x = solve_two_iterations(solver, lambd)
        xs.append(x)

    bin_and_plot(xs, labels, fname="gridsearch_lambda_x_binned")


def main():
    stringart_directory: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory: Path = stringart_directory.parent.resolve()

    image_cropped = crop_image(image, crop_mode)
    solver = Solver(image_cropped, crop_mode, number_of_pegs=number_of_pegs, rasterization=rasterization)

    Benchmark.initialize_metadata(directory)
    benchmark = Benchmark(image, crop_mode, number_of_pegs, rasterization)

    run_benchmarks(benchmark, solver)
    solve_plot_x(solver)


if __name__ == "__main__":
    main()
