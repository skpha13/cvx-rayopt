import numpy as np
import scipy
from matplotlib import pyplot as plt
from stringart.line_algorithms.matrix import MatrixGenerator
from stringart.misc.least_squares_rounding_trial import compute_solution
from stringart.utils.image import ImageWrapper
from stringart.utils.types import CropMode, MatrixRepresentation

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
number_of_pegs = 100
matrix_representation: MatrixRepresentation = "sparse"


def linear_least_squares(
    src: np.ndarray,
    shape: tuple[int, ...],
    number_of_pegs: int = 100,
    crop_mode: CropMode = "center",
    matrix_representation: MatrixRepresentation = "sparse",
) -> tuple[np.ndarray, np.ndarray]:
    A, _ = MatrixGenerator.compute_matrix(shape, number_of_pegs, crop_mode, matrix_representation, rasterized=True)
    b: np.ndarray = ImageWrapper.flatten_image(src)

    optimize_results = scipy.optimize.lsq_linear(A, b, bounds=(0, np.inf))

    return A, optimize_results.x


def main():
    src = ImageWrapper.histogram_equalization(image)
    A, x = linear_least_squares(src, shape, number_of_pegs, crop_mode, matrix_representation)
    solution = compute_solution(A, x)

    plt.imshow(solution, cmap="grey")
    plt.axis("off")
    plt.show()
    plt.imsave("../../outputs/misc/preprocess_xiaolin_wu_lena.png", solution, cmap="grey")


if __name__ == "__main__":
    main()
