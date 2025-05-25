import time

import matplotlib.pyplot as plt
import numpy as np
from stringart.optimize.downsampling import UDSLoss
from stringart.solver import Solver
from stringart.utils.image import ImageWrapper, crop_image
from stringart.utils.time_memory_format import convert_monotonic_time, format_time
from stringart.utils.types import CropMode, MatchingPursuitMethod, MatrixRepresentation, Rasterization

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
matrix_representation: MatrixRepresentation = "sparse"
number_of_pegs: int = 100
rasterization: Rasterization = "xiaolin-wu"

number_of_lines: int = 1000
mp_method: MatchingPursuitMethod = "orthogonal"


def main():
    start = time.monotonic()
    residual_fn = UDSLoss(image, crop_mode, number_of_pegs, rasterization, block_size=2)
    elapsed = time.monotonic() - start
    print(f"Init: {format_time(convert_monotonic_time(elapsed))}")

    image_cropped = crop_image(image, crop_mode)

    solver = Solver(image_cropped, crop_mode, number_of_pegs=number_of_pegs, rasterization=rasterization)
    A, x, _ = solver.ls(matrix_representation)
    base_residual = np.linalg.norm(solver.b - A @ x)
    print(f"Base Residual: {base_residual:.6f}")

    base = solver.compute_solution(A, x)
    plt.imshow(base, cmap="grey")
    plt.axis("off")
    plt.show()

    start = time.monotonic()
    solution, residual = residual_fn(x)
    elapsed = time.monotonic() - start
    print(f"UDS  Residual: {residual:.6f}")
    print(f"Call: {format_time(convert_monotonic_time(elapsed))}")

    plt.imshow(solution, cmap="grey")
    plt.axis("off")
    plt.show()
    plt.imsave("../../outputs/misc/downsamples_ls.png", solution, cmap="grey")


if __name__ == "__main__":
    main()
