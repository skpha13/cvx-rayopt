import matplotlib.pyplot as plt
import numpy as np
from stringart.solver import Solver
from stringart.utils.image import ImageWrapper, crop_image

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
image_mode = "center"


def compute_solution(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    l = 1000
    value = x[np.argsort(x)[-l]]

    xp = x.copy()
    xp[xp < value] = 0
    xp = np.clip(xp, a_min=0, a_max=1)

    solution = A @ xp
    solution = np.clip(np.reshape(solution, shape=shape), a_min=0, a_max=1)
    solution = 1 - solution
    solution = np.multiply(solution, 255).astype(np.uint8)
    solution = crop_image(solution, image_mode)

    return solution


def main():
    solver = Solver(image, image_mode, number_of_pegs=300)
    A, x = solver.least_squares("sparse")
    solution = compute_solution(A, x)

    plt.imshow(solution, cmap="grey")
    plt.show()
    plt.imsave("../../outputs/misc/least_squares_lena_rounding.png", solution, cmap="grey")


if __name__ == "__main__":
    main()
