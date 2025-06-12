import os
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from stringart.solver import Solver
from stringart.utils.image import ImageWrapper, crop_image, masked_rmse
from stringart.utils.perf_analyzer import Benchmark, BenchmarkResult
from stringart.utils.types import CropMode, Rasterization

crop_mode: CropMode = "center"
number_of_pegs = 256
rasterization: Rasterization = "xiaolin-wu"
block_size: int = 8

image_dir = Path("../../imgs/radon")
image_paths = sorted([file for file in image_dir.iterdir() if file.is_file()])
images = [
    ImageWrapper.apply_alpha_map_bw_to_rgba(
        crop_image(ImageWrapper.read_bw(path, inverted=False), crop_mode),
        ImageWrapper.alpha_map(crop_image(ImageWrapper.read_bw(path, inverted=True), crop_mode), crop_mode),
    )
    for path in image_paths
]


def plot_input_output_grid(images, solutions, labels=None, max_cols=5):
    num_images = len(images)
    num_rows = (num_images + max_cols - 1) // max_cols

    image_width = 3
    image_height = 3
    text_height = 0.5

    fig_width = max_cols * image_width
    fig_height = num_rows * (2 * image_height + text_height)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        num_rows * 3,
        max_cols,
        figure=fig,
        height_ratios=[image_height, image_height, text_height] * num_rows,
        hspace=0,
        wspace=0.05,
    )
    axs = np.empty((num_rows * 3, max_cols), dtype=object)

    for row in range(num_rows * 3):
        for col in range(max_cols):
            axs[row, col] = fig.add_subplot(gs[row, col])
            axs[row, col].set_axis_off()

    # plot each pair
    for idx, (input_img, output_img) in enumerate(zip(images, solutions)):
        col = idx % max_cols
        row_group = (idx // max_cols) * 3

        axs[row_group, col].imshow(input_img, cmap="gray")
        axs[row_group + 1, col].imshow(output_img, cmap="gray")

        if labels:
            axs[row_group + 2, col].text(
                0.5,
                0.5,
                labels[idx],
                ha="center",
                va="center",
                transform=axs[row_group + 2, col].transAxes,
                fontsize=12,
                wrap=True,
            )

    # hide any remaining axes
    total_slots = num_rows * max_cols
    for empty_idx in range(num_images, total_slots):
        col = empty_idx % max_cols
        row_group = (empty_idx // max_cols) * 3
        for r in range(3):
            axs[row_group + r, col].axis("off")

    fig.subplots_adjust(wspace=0, hspace=0.01)
    fig.savefig(
        "../../outputs/experiments/radon_quantitative_test.pdf", format="pdf", bbox_inches="tight", pad_inches=0
    )


def run_benchmark(image_path: Path) -> BenchmarkResult:
    image = ImageWrapper.read_bw(image_path)
    image_cropped = crop_image(image, crop_mode)

    solver = Solver(image_cropped, crop_mode, number_of_pegs, rasterization, block_size)
    benchmark = Benchmark(image, crop_mode, number_of_pegs, rasterization, block_size=block_size)

    radon_results = benchmark.run_benchmark(
        solver.radon,
        uds=True,
        patience=10,
    )

    return radon_results


def run_benchmarks() -> list[BenchmarkResult]:
    results = []
    for path in image_paths:
        result = run_benchmark(path)
        results.append(result)

    return results


def main():
    stringart_directory: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory: Path = stringart_directory.parent.resolve()

    Benchmark.initialize_metadata(directory)
    if not os.path.exists("../../benchmarks/radon_quantitative.json"):
        results = run_benchmarks()
        Benchmark.save_benchmarks(results, "radon_quantitative")
    else:
        results = Benchmark.load_benchmarks("radon_quantitative")

    letter_labels = [f"({letter})" for letter in list(string.ascii_lowercase[: len(image_paths)])]
    solutions = [results.output_image for results in results]
    plot_input_output_grid(images, solutions, letter_labels)

    for label, result, input_image in zip(letter_labels, results, images):
        number_of_lines = (result.x == 1).sum()
        print(f"Label: {label}, Selected Lines: {number_of_lines}")

        input_image = ImageWrapper.scale_image(input_image)
        output_image = ImageWrapper.scale_image(result.output_image)

        output_image = ImageWrapper.apply_alpha_map_bw_to_rgba(
            output_image, ImageWrapper.alpha_map(output_image, crop_mode)
        )
        rms = masked_rmse(input_image, output_image)
        print(f"Label: {label}, RMS: {rms}")

    benchmark = Benchmark(
        ImageWrapper.read_bw(image_paths[0]), crop_mode, number_of_pegs, rasterization, block_size=block_size
    )
    benchmark.run_analysis(results, 1 - ImageWrapper.read_bw(image_paths[0]), "radon_quantitative_analysis")


if __name__ == "__main__":
    main()
