import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from stringart.optimize.downsampling import UDSLoss
from stringart.solver import Solver
from stringart.utils.image import ImageWrapper, crop_image
from stringart.utils.perf_analyzer import Benchmark
from stringart.utils.types import CropMode, MatrixRepresentation, Rasterization

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
number_of_pegs = 128
matrix_representation: MatrixRepresentation = "sparse"
rasterization: Rasterization = "xiaolin-wu"
number_of_lines = 1000


def udps(x: np.ndarray, block_size: int = 2) -> tuple[np.floating, np.ndarray]:
    residual_fn = UDSLoss(image, crop_mode, number_of_pegs, rasterization, block_size=block_size)
    return residual_fn(x)


def lls_run(solver: Solver):
    A, x, _ = solver.lls(matrix_representation)

    k = number_of_lines
    value = x[np.argsort(x)[-k]]

    xp = x.copy()
    xp[xp < value] = 0
    xp[xp >= value] = 1

    block_sizes = [2, 4, 8, 16]
    residuals = []
    solutions = []

    for block_size in block_sizes:
        residual, solution = udps(xp, block_size)
        residuals.append(residual)
        solutions.append(solution)

    for block_size, residual in zip(block_sizes, residuals):
        print(f"Residual block_size={block_size}: {residual:.6f}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, sol, size in zip(axes, solutions, block_sizes):
        ax.imshow(sol, cmap="gray")
        ax.set_title(f"Block size = {size}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    stringart_directory: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory: Path = stringart_directory.parent.resolve()
    Benchmark.initialize_metadata(directory)
    benchmark = Benchmark(image, crop_mode, number_of_pegs, rasterization)

    image_cropped = crop_image(image, crop_mode)
    solver = Solver(image_cropped, crop_mode, number_of_pegs=number_of_pegs, rasterization=rasterization)
    residual_fn = UDSLoss(image, crop_mode, number_of_pegs, rasterization, block_size=2)

    lls_run(solver)
