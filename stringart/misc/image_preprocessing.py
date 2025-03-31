from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from stringart.misc.least_squares_rounding_trial import compute_solution
from stringart.misc.linear_least_squares import linear_least_squares
from stringart.utils.image import ImageWrapper
from stringart.utils.types import CropMode, MatrixRepresentation

image_path = "../../imgs/lena.png"
image_noninverted = ImageWrapper.read_bw(image_path, inverted=False)
cmap = "grey"

image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
number_of_pegs = 100
matrix_representation: MatrixRepresentation = "sparse"


def save_img(img: np.ndarray, name: str, cmap: str = "grey") -> None:
    plt.imshow(img, cmap=cmap)
    plt.imsave(f"../../outputs/preprocess/{name}.png", img, cmap=cmap)


def plot_preprocess(src: np.ndarray, dst: np.ndarray, title: str, fname: str) -> None:
    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    fig.suptitle(title)

    axs[0].set_title("Input Image")
    axs[0].imshow(src, cmap=cmap)
    axs[0].axis("off")

    axs[1].set_title("Output Image")
    axs[1].imshow(dst, cmap=cmap)
    axs[1].axis("off")

    fig.show()
    fig.savefig(fname)


def process_image(
    image: np.ndarray, preprocess_func: Callable, shape, number_of_pegs, crop_mode, matrix_representation
):
    processed_image = preprocess_func(image)
    A, x = linear_least_squares(processed_image, shape, number_of_pegs, crop_mode, matrix_representation)
    return compute_solution(A, x)


def identity(src: np.ndarray) -> np.ndarray:
    return src


def compute_solution_parallel(src, shape, number_of_pegs, crop_mode, matrix_representation):
    with ProcessPoolExecutor() as executor:
        futures = {
            "raw": executor.submit(
                process_image,
                src,
                identity,
                shape,
                number_of_pegs,
                crop_mode,
                matrix_representation,
            ),
            "histogram_equalization": executor.submit(
                process_image,
                src,
                ImageWrapper.histogram_equalization,
                shape,
                number_of_pegs,
                crop_mode,
                matrix_representation,
            ),
            "grayscale_quantization": executor.submit(
                process_image,
                src,
                ImageWrapper.grayscale_quantization,
                shape,
                number_of_pegs,
                crop_mode,
                matrix_representation,
            ),
        }

        # Collect the results
        results = {name: future.result() for name, future in futures.items()}

    # Extract results
    solution_raw = results["raw"]
    solution_histogram_equalization = results["histogram_equalization"]
    solution_grayscale_quantization = results["grayscale_quantization"]

    return solution_raw, solution_histogram_equalization, solution_grayscale_quantization


def compute_plot_preprocessed(src: np.ndarray) -> None:
    solution_raw, solution_histogram_equalization, solution_grayscale_quantization = compute_solution_parallel(
        src,
        shape,
        number_of_pegs,
        crop_mode,
        matrix_representation,
    )

    fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(12, 4))
    fig.suptitle("Preprocess Effects using Linear Least Squares")

    axs[0].set_title("Input Image")
    axs[0].imshow(image_noninverted, cmap=cmap)
    axs[0].axis("off")

    axs[1].set_title("Raw")
    axs[1].imshow(solution_raw, cmap=cmap)
    axs[1].axis("off")

    axs[2].set_title("Histogram Equalization")
    axs[2].imshow(solution_histogram_equalization, cmap=cmap)
    axs[2].axis("off")

    axs[3].set_title("Grayscale Quantization 2 Levels")
    axs[3].imshow(solution_grayscale_quantization, cmap=cmap)
    axs[3].axis("off")

    fig.show()
    fig.savefig("../../outputs/preprocess/image_processing_stages.png")


def main():
    src = image_noninverted.copy()

    dst = ImageWrapper.histogram_equalization(src)
    plot_preprocess(src, dst, "Histogram Equalization", "../../outputs/misc/histogram_equalization_preprocess.png")
    save_img(dst, "histogram_equalization_preprocess")

    dst = ImageWrapper.grayscale_quantization(src)
    # fmt: off
    plot_preprocess(src, dst,"Grayscale Quantization 2 Levels","../../outputs/misc/grayscale_quantization_preprocess.png",)
    # fmt: on
    save_img(dst, "grayscale_quantization_2_levels")

    compute_plot_preprocessed(image)


if __name__ == "__main__":
    main()
