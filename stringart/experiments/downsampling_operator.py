import matplotlib.pyplot as plt
import numpy as np
from stringart.optimize.downsampling import UDSLoss
from stringart.solver import Solver
from stringart.utils.image import ImageWrapper, crop_image
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
    residual_fn = UDSLoss(image, matrix_representation, crop_mode, number_of_pegs, rasterization)
    image_cropped = crop_image(image, crop_mode)

    solver = Solver(image_cropped, crop_mode, number_of_pegs=number_of_pegs, rasterization=rasterization)
    A, x, _ = solver.ls(matrix_representation)
    base_residual = np.linalg.norm(solver.b - A @ x)
    print(f"Base Residual: {base_residual:.6f}")

    base = solver.compute_solution(A, x)
    plt.imshow(base, cmap="grey")
    plt.axis("off")
    plt.show()

    solution, residual = residual_fn(x, shape, block_size=2)
    print(f"UDS  Residual: {residual:.6f}")

    plt.imshow(solution, cmap="grey")
    plt.axis("off")
    plt.show()
    plt.imsave("../../outputs/misc/downsamples_ls.png", solution, cmap="grey")


if __name__ == "__main__":
    main()
