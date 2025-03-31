from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from stringart.misc.least_squares_rounding_trial import compute_solution
from stringart.misc.linear_least_squares import linear_least_squares
from stringart.utils.image import ImageWrapper
from stringart.utils.performance_analysis import prepare_diff_images
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
                np.copy(src),
                identity,
                shape,
                number_of_pegs,
                crop_mode,
                matrix_representation,
            ),
            "histogram_equalization": executor.submit(
                process_image,
                np.copy(src),
                ImageWrapper.histogram_equalization,
                shape,
                number_of_pegs,
                crop_mode,
                matrix_representation,
            ),
            "grayscale_quantization": executor.submit(
                process_image,
                np.copy(src),
                ImageWrapper.grayscale_quantization,
                shape,
                number_of_pegs,
                crop_mode,
                matrix_representation,
            ),
        }

        # Collect the results
        results = {name: np.array(future.result()) for name, future in futures.items()}

    # Extract results
    solution_raw = results["raw"]
    solution_histogram_equalization = results["histogram_equalization"]
    solution_grayscale_quantization = results["grayscale_quantization"]

    return solution_raw, solution_histogram_equalization, solution_grayscale_quantization


def plot_preprocessed(raw: np.ndarray, histogram_equalization: np.ndarray, grayscale_quantization: np.ndarray) -> None:
    fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(12, 4))
    fig.suptitle("Preprocess Effects using Linear Least Squares")

    axs[0].set_title("Input Image")
    axs[0].imshow(image_noninverted, cmap=cmap)
    axs[0].axis("off")

    axs[1].set_title("Raw")
    axs[1].imshow(raw, cmap=cmap)
    axs[1].axis("off")

    axs[2].set_title("Histogram Equalization")
    axs[2].imshow(histogram_equalization, cmap=cmap)
    axs[2].axis("off")

    axs[3].set_title("Grayscale Quantization 2 Levels")
    axs[3].imshow(grayscale_quantization, cmap=cmap)
    axs[3].axis("off")

    fig.show()
    fig.savefig("../../outputs/preprocess/image_processing_stages.png")


def plot_diff_preprocess_methods(
    raw: np.ndarray, histogram_equalization: np.ndarray, grayscale_quantization: np.ndarray
) -> None:
    raw_hist_diff = histogram_equalization - raw
    raw_gray_diff = grayscale_quantization - raw
    raw_hist_diff_prepared, raw_gray_diff_prepared = prepare_diff_images([raw_hist_diff, raw_gray_diff], crop_mode)

    cmap = "plasma"
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(9, 4.5))
    fig.suptitle("Diff Analysis on Preprocess Methods")

    axs[0].set_title("Histogram Equalization")
    axs[0].imshow(raw_hist_diff_prepared, cmap=cmap)
    axs[0].axis("off")

    axs[1].set_title("Grayscale Quantization")
    axs[1].imshow(raw_gray_diff_prepared, cmap=cmap)
    axs[1].axis("off")

    fig.show()
    fig.savefig("../../outputs/preprocess/diff_analysis_on_preprocess_methods.png")


def plot_fft_preprocess_methods(
    raw: np.ndarray, histogram_equalization: np.ndarray, grayscale_quantization: np.ndarray
) -> None:
    raw_fft_spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(raw))) + 1)
    hist_fft_spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(histogram_equalization))) + 1)
    gray_fft_spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(grayscale_quantization))) + 1)

    cmap = "plasma"
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(9, 3.5))
    fig.suptitle("FFT Analysis on Preprocess Methods")

    axs[0].set_title("Raw")
    axs[0].imshow(raw_fft_spectrum, cmap=cmap)
    axs[0].axis("off")

    axs[1].set_title("Histogram Equalization")
    axs[1].imshow(hist_fft_spectrum, cmap=cmap)
    axs[1].axis("off")

    axs[2].set_title("Grayscale Quantization")
    axs[2].imshow(gray_fft_spectrum, cmap=cmap)
    axs[2].axis("off")

    fig.show()
    fig.savefig("../../outputs/preprocess/fft_analysis_on_preprocess_methods.png")


def plot_overlay_analysis(
    raw: np.ndarray, histogram_equalization: np.ndarray, grayscale_quantization: np.ndarray
) -> None:
    def overlay(img1: np.ndarray, img2: np.ndarray, amount: float = 0.5) -> np.ndarray:
        # the function x^2 compresses values closer to 0 more aggressive, while preserving 1 as it is
        img1 = img1**2
        img2 = img2**2

        img1_colored = np.ones((img1.shape[0], img1.shape[1], 3))
        img1_colored[..., 0] = 1  # Red channel
        img1_colored[..., 1] = img1  # Green channel
        img1_colored[..., 2] = img1  # Blue channel

        img2_colored = np.ones((img2.shape[0], img2.shape[1], 3))
        img2_colored[..., 0] = img2  # Red channel
        img2_colored[..., 1] = img2  # Green channel
        img2_colored[..., 2] = 1  # Blue channel

        return amount * img1_colored + (1 - amount) * img2_colored

    plt.imshow(raw, cmap="grey")
    plt.show()

    overlay_hist = overlay(raw, histogram_equalization)
    overlay_gray = overlay(raw, grayscale_quantization)

    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(9, 4.5))
    fig.suptitle("Overlay Analysis on Preprocess Methods")

    axs[0].set_title("Histogram Equalization")
    axs[0].imshow(overlay_hist)
    axs[0].axis("off")

    axs[1].set_title("Grayscale Quantization")
    axs[1].imshow(overlay_gray)
    axs[1].axis("off")

    fig.show()
    fig.savefig("../../outputs/preprocess/overlay_analysis_on_preprocess_methods.png")


def main():
    src = image_noninverted.copy()

    dst = ImageWrapper.histogram_equalization(src)
    plot_preprocess(src, dst, "Histogram Equalization", "../../outputs/misc/histogram_equalization_preprocess.png")
    save_img(dst, "histogram_equalization_preprocess")

    dst = ImageWrapper.grayscale_quantization(src)
    # fmt: off
    plot_preprocess(src, dst,"Grayscale Quantization 2 Levels","../../outputs/misc/grayscale_quantization_preprocess.png")
    # fmt: on
    save_img(dst, "grayscale_quantization_2_levels")

    raw, histogram_equalization, grayscale_quantization = compute_solution_parallel(
        np.copy(image),
        shape,
        number_of_pegs,
        crop_mode,
        matrix_representation,
    )

    # scaling back down to [0, 1]
    raw = ImageWrapper.scale_image(raw)
    histogram_equalization = ImageWrapper.scale_image(histogram_equalization)
    grayscale_quantization = ImageWrapper.scale_image(grayscale_quantization)

    plot_preprocessed(raw, histogram_equalization, grayscale_quantization)
    plot_diff_preprocess_methods(raw, histogram_equalization, grayscale_quantization)
    plot_overlay_analysis(raw, histogram_equalization, grayscale_quantization)
    plot_fft_preprocess_methods(raw, histogram_equalization, grayscale_quantization)


if __name__ == "__main__":
    main()
