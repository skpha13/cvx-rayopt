from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from stringart.experiments.least_squares_rounding import compute_solution
from stringart.experiments.linear_least_squares import linear_least_squares
from stringart.utils.image import ImageWrapper
from stringart.utils.perf_analyzer import prepare_diff_images
from stringart.utils.types import CropMode, MatrixRepresentation

image_path = "../../imgs/lena.png"
image_noninverted = ImageWrapper.read_bw(image_path, inverted=False)
cmap = "gray"

image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
number_of_pegs = 100
matrix_representation: MatrixRepresentation = "sparse"


def save_img(img: np.ndarray, name: str, cmap: str = "gray") -> None:
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
    A, x, _ = linear_least_squares(processed_image, shape, number_of_pegs, crop_mode, matrix_representation)
    return compute_solution(A, x, shape=shape, l=4950)


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
            "sketch_effect": executor.submit(
                process_image,
                np.copy(src),
                ImageWrapper.sketch_effect,
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
    solution_sketch_effect = results["sketch_effect"]

    return solution_raw, solution_histogram_equalization, solution_grayscale_quantization, solution_sketch_effect


def plot_preprocessed(
    input_image: np.ndarray,
    raw: np.ndarray,
    histogram_equalization: np.ndarray,
    grayscale_quantization: np.ndarray,
    sketch_effect: np.ndarray,
    output_path: str | Path | None = None,
) -> None:
    fig, axs = plt.subplots(1, 5, constrained_layout=True, figsize=(20, 4))
    fig.suptitle("Preprocess Effects using Linear Least Squares")

    axs[0].set_title("Input Image")
    axs[0].imshow(input_image, cmap=cmap)
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

    axs[4].set_title("Sketch Effect")
    axs[4].imshow(sketch_effect, cmap=cmap)
    axs[4].axis("off")

    fig.show()

    path = output_path if output_path is not None else "../../outputs/preprocess/image_processing_stages.png"
    fig.savefig(path)


def plot_diff_preprocess_methods(
    raw: np.ndarray,
    histogram_equalization: np.ndarray,
    grayscale_quantization: np.ndarray,
    sketch_effect: np.ndarray,
    output_path: str | Path | None = None,
) -> None:
    raw_hist_diff = np.abs(histogram_equalization - raw)
    raw_gray_diff = np.abs(grayscale_quantization - raw)
    raw_sketch_diff = np.abs(sketch_effect - raw)
    raw_hist_diff_prepared, raw_gray_diff_prepared, raw_sketch_diff_prepared = prepare_diff_images(
        [raw_hist_diff, raw_gray_diff, raw_sketch_diff], crop_mode
    )

    cmap = "plasma"
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 4))
    fig.suptitle("Diff Analysis on Preprocess Methods")

    axs[0].set_title("Histogram Equalization")
    axs[0].imshow(raw_hist_diff_prepared, cmap=cmap)
    axs[0].axis("off")

    axs[1].set_title("Grayscale Quantization")
    axs[1].imshow(raw_gray_diff_prepared, cmap=cmap)
    axs[1].axis("off")

    axs[2].set_title("Sketch Effect")
    axs[2].imshow(raw_sketch_diff_prepared, cmap=cmap)
    axs[2].axis("off")

    fig.show()

    path = (
        output_path if output_path is not None else "../../outputs/preprocess/diff_analysis_on_preprocess_methods.png"
    )
    fig.savefig(path)


def plot_fft_preprocess_methods(
    raw: np.ndarray,
    histogram_equalization: np.ndarray,
    grayscale_quantization: np.ndarray,
    sketch_effect: np.ndarray,
    output_path: str | Path | None = None,
) -> None:
    raw_fft_spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(raw))) + 1)
    hist_fft_spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(histogram_equalization))) + 1)
    gray_fft_spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(grayscale_quantization))) + 1)
    sketch_fft_spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(sketch_effect))) + 1)

    cmap = "plasma"
    fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(16, 4))
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

    axs[3].set_title("Sketch Effect")
    axs[3].imshow(sketch_fft_spectrum, cmap=cmap)
    axs[3].axis("off")

    fig.show()

    path = output_path if output_path is not None else "../../outputs/preprocess/fft_analysis_on_preprocess_methods.png"
    fig.savefig(path)


def plot_overlay_analysis(
    raw: np.ndarray,
    histogram_equalization: np.ndarray,
    grayscale_quantization: np.ndarray,
    sketch_effect: np.ndarray,
    output_path: str | Path | None = None,
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

    overlay_hist = overlay(raw, histogram_equalization)
    overlay_gray = overlay(raw, grayscale_quantization)
    overlay_sketch = overlay(raw, sketch_effect)

    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(12, 4))
    fig.suptitle("Overlay Analysis on Preprocess Methods")

    axs[0].set_title("Histogram Equalization")
    axs[0].imshow(overlay_hist)
    axs[0].axis("off")

    axs[1].set_title("Grayscale Quantization")
    axs[1].imshow(overlay_gray)
    axs[1].axis("off")

    axs[2].set_title("Sketch Effect")
    axs[2].imshow(overlay_sketch)
    axs[2].axis("off")

    fig.show()

    path = (
        output_path
        if output_path is not None
        else "../../outputs/preprocess/overlay_analysis_on_preprocess_methods.png"
    )
    fig.savefig(path)


def main():
    image_paths = [
        "../../imgs/lena.png",
        "../../imgs/airplane.png",
        "../../imgs/butterfly.png",
        "../../imgs/tank.png",
        "../../imgs/teddybear.png",
    ]

    for image_path in image_paths:
        img_name = Path(image_path).stem
        output_dir = Path(f"../../outputs/preprocess/{img_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        image_noninverted = ImageWrapper.read_bw(image_path, inverted=False)
        image = ImageWrapper.read_bw(image_path)
        shape = image.shape

        src = image_noninverted.copy()

        for name, func in [
            ("histogram_equalization", ImageWrapper.histogram_equalization),
            ("grayscale_quantization_2_levels", ImageWrapper.grayscale_quantization),
            ("sketch_effect", ImageWrapper.sketch_effect),
        ]:
            dst = func(src)
            plot_preprocess(src, dst, name.replace("_", " ").title(), str(output_dir / f"{name}_preview.png"))
            save_img(dst, f"{img_name}/{name}")

        raw, histogram_equalization, grayscale_quantization, sketch_effect = compute_solution_parallel(
            np.copy(image),
            shape,
            number_of_pegs,
            crop_mode,
            matrix_representation,
        )

        raw = ImageWrapper.scale_image(raw)
        histogram_equalization = ImageWrapper.scale_image(histogram_equalization)
        grayscale_quantization = ImageWrapper.scale_image(grayscale_quantization)
        sketch_effect = ImageWrapper.scale_image(sketch_effect)

        plot_preprocessed(
            image_noninverted,
            raw,
            histogram_equalization,
            grayscale_quantization,
            sketch_effect,
            output_dir / "preprocessed_results.png",
        )
        plot_diff_preprocess_methods(
            raw,
            histogram_equalization,
            grayscale_quantization,
            sketch_effect,
            output_dir / "diff_analysis.png",
        )
        plot_overlay_analysis(
            raw,
            histogram_equalization,
            grayscale_quantization,
            sketch_effect,
            output_dir / "overlay_analysis.png",
        )
        plot_fft_preprocess_methods(
            raw,
            histogram_equalization,
            grayscale_quantization,
            sketch_effect,
            output_dir / "fft_analysis.png",
        )


if __name__ == "__main__":
    main()
