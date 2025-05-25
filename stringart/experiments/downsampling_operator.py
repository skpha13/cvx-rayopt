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
mp_method: MatchingPursuitMethod = "greedy"


def plot_image(image: np.ndarray, fname: str):
    plt.imshow(image, cmap="grey")
    plt.axis("off")
    plt.show()
    plt.imsave(f"../../outputs/misc/{fname}", image, cmap="grey")


def dls(solver: Solver, residual_fn: UDSLoss):
    """Downsampled Least Squares"""
    A, x, _ = solver.ls(matrix_representation)
    base_residual = np.linalg.norm(solver.b - A @ x)
    print(f"Base Residual: {base_residual:.6f}")

    base = solver.compute_solution(A, x)
    plot_image(base, "base_ls.png")

    start = time.monotonic()
    residual, solution = residual_fn(x)
    elapsed = time.monotonic() - start
    print(f"UDS  Residual: {residual:.6f}")
    print(f"Call: {format_time(convert_monotonic_time(elapsed))}")

    plot_image(solution, fname="downsampled_ls.png")


def dmp(solver: Solver, residual_fn: UDSLoss):
    """Downsampled Matching Pursuit"""
    A, x, residuals = solver.mp(number_of_lines, mp_method, selector_type="random")

    start = time.monotonic()
    residual, solution = residual_fn(x)
    elapsed = time.monotonic() - start
    print(f"UDS  Residual: {residual:.6f}")
    print(f"Call: {format_time(convert_monotonic_time(elapsed))}")

    plot_image(solution, "downsampled_mp.png")


def main():
    image_cropped = crop_image(image, crop_mode)
    solver = Solver(image_cropped, crop_mode, number_of_pegs=number_of_pegs, rasterization=rasterization)

    start = time.monotonic()
    residual_fn = UDSLoss(image, crop_mode, number_of_pegs, rasterization, block_size=2)
    elapsed = time.monotonic() - start
    print(f"Init: {format_time(convert_monotonic_time(elapsed))}")

    # dls(solver, residual_fn)
    dmp(solver, residual_fn)


if __name__ == "__main__":
    main()
