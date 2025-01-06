import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsqr
from skimage import io
from stringart.line_algorithms.matrix import MatrixGenerator
from stringart.utils.image import ImageWrapper, crop_image
from stringart.utils.types import Method, Mode

image = ImageWrapper.read_bw("../../imgs/lena.png")
mode: Mode = "center"
method: Method = "sparse"
shape = image.shape
b = ImageWrapper.flatten_image(image)


def display_line_console(A: np.ndarray, pegs: np.ndarray) -> None:
    first_line = A[:, 2]
    matrix = np.reshape(first_line, shape=shape)
    pegs_arr = [[point.y, point.x] for point in pegs]

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if [i, j] in pegs_arr:
                print("*", end=" ")
                continue
            print("." if matrix[i][j] == 0 else "#", end=" ")
        print()


def matrix_solver() -> None:
    A, pegs = MatrixGenerator.compute_matrix(shape, 100, mode, method)

    x = None
    if method == "dense":
        x, _, _, _ = lstsq(A, b)
    elif method == "sparse":
        x = lsqr(A, b)[0]

    solution = A @ x
    solution = np.clip(np.reshape(solution, shape=shape), a_min=0, a_max=1)
    solution = np.multiply(solution, 255).astype(np.uint8)
    solution = crop_image(solution, mode)

    io.imsave("../../outputs/lena_stringart_sparse.png", solution)
    io.imshow(solution)
    plt.show()


if __name__ == "__main__":
    matrix_solver()
