import os
from pathlib import Path

import numpy as np
import pulp
from stringart.line_algorithms.matrix import MatrixGenerator
from stringart.utils.image import ImageWrapper
from stringart.utils.perf_analyzer import Benchmark
from stringart.utils.types import CropMode, MatrixRepresentation

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
# number_of_pegs = 100 -> 6513.97 seconds
number_of_pegs = 30


def linear_programming(
    shape: tuple[int, ...],
    number_of_pegs: int = 100,
    image_mode: CropMode = "center",
    method: MatrixRepresentation = "sparse",
) -> tuple[np.ndarray, np.ndarray]:
    A, _ = MatrixGenerator.compute_matrix(shape, number_of_pegs, image_mode, method)
    b: np.ndarray = ImageWrapper.flatten_image(image)

    model = pulp.LpProblem("Sparse_Regression_CSR_Positive_X", pulp.LpMinimize)

    # decision variables
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0, cat="Continuous") for j in range(A.shape[1])}
    # residuals
    r = {i: pulp.LpVariable(f"r_{i}", lowBound=0, cat="Continuous") for i in range(A.shape[0])}

    # objective function
    model += pulp.lpSum(r[i] for i in range(A.shape[0]))

    # residual constraints
    for i in range(A.shape[0]):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]

        row_data = A.data[row_start:row_end]
        row_indices = A.indices[row_start:row_end]

        # https://en.wikipedia.org/wiki/Least_absolute_deviations#Solution_using_linear_programming
        model += r[i] >= pulp.lpSum(row_data[k] * x[row_indices[k]] for k in range(len(row_data))) - b[i]
        model += r[i] >= b[i] - pulp.lpSum(row_data[k] * x[row_indices[k]] for k in range(len(row_data)))

    model.solve(pulp.PULP_CBC_CMD(msg=True))

    result = np.array([pulp.value(x[j]) if pulp.value(x[j]) >= 1e-6 else 0.0 for j in range(A.shape[1])])

    return A, result


def main():
    stringart_directory: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory: Path = stringart_directory.parent.resolve()

    Benchmark.initialize_metadata(directory)
    benchmark = Benchmark(image, crop_mode, number_of_pegs)
    benchmarks_result = benchmark.run_benchmark(
        linear_programming,
        shape=shape,
        number_of_pegs=number_of_pegs,
        crop_mode=crop_mode,
        matrix_representation="sparse",
    )

    benchmark.save_benchmarks([benchmarks_result], "linear_programming")


if __name__ == "__main__":
    main()
