import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_blocks
from stringart.line_algorithms.matrix import MatrixGenerator
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


def downsample(x: np.ndarray, block_size: int = 8) -> np.ndarray:
    upsample_shape = (shape[0] * block_size, shape[1] * block_size)
    A_upsampled, _ = MatrixGenerator.compute_matrix(
        upsample_shape, number_of_pegs, crop_mode, matrix_representation, rasterization
    )

    solution = A_upsampled @ x
    solution = np.clip(np.reshape(solution, upsample_shape), a_min=0, a_max=1)

    plt.imshow(1 - solution, cmap="grey")
    plt.axis("off")
    plt.show()

    blocks = view_as_blocks(solution, (block_size, block_size))
    ds = blocks.mean(axis=(2, 3))

    solution = 1 - ds
    solution = np.multiply(solution, 255).astype(np.uint8)

    return solution


def main():
    image_cropped = crop_image(image, crop_mode)
    solver = Solver(image_cropped, crop_mode, number_of_pegs=number_of_pegs, rasterization=rasterization)
    A, x, _ = solver.least_squares(matrix_representation)

    base = solver.compute_solution(A, x)
    plt.imshow(base, cmap="grey")
    plt.axis("off")
    plt.show()

    solution = downsample(x, block_size=2)

    plt.imshow(solution, cmap="grey")
    plt.axis("off")
    plt.show()
    plt.imsave("../../outputs/misc/downsamples_ls.png", solution, cmap="grey")


if __name__ == "__main__":
    main()
