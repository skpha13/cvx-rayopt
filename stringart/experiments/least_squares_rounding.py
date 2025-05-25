from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from stringart.solver import Solver
from stringart.utils.image import ImageWrapper, crop_image
from stringart.utils.types import CropMode

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"


def compute_solution(A: np.ndarray, x: np.ndarray, l: int = 1000, shape: Tuple[int, ...] = (330, 330)) -> np.ndarray:
    value = x[np.argsort(x)[-l]]

    xp = x.copy()
    xp[xp < value] = 0
    xp = np.clip(xp, a_min=0, a_max=1)

    solution = A @ xp
    solution = np.clip(np.reshape(solution, shape=shape), a_min=0, a_max=1)
    solution = 1 - solution
    solution = np.multiply(solution, 255).astype(np.uint8)

    return solution


def main():
    image_cropped = crop_image(image, crop_mode)
    solver = Solver(image_cropped, crop_mode, number_of_pegs=300)
    A, x, _ = solver.ls("sparse")
    solution = compute_solution(A, x)

    plt.imshow(solution, cmap="grey")
    plt.show()
    plt.imsave("../../outputs/misc/least_squares_lena_rounding.png", solution, cmap="grey")


if __name__ == "__main__":
    main()
