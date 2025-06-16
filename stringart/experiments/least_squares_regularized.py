import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from stringart.solver import Solver
from stringart.utils.image import ImageWrapper, crop_image
from stringart.utils.perf_analyzer import Benchmark
from stringart.utils.types import CropMode, MatrixRepresentation, Rasterization

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
number_of_pegs = 100
matrix_representation: MatrixRepresentation = "sparse"
rasterization: Rasterization = "xiaolin-wu"
lambd = 5000


def solve():
    print("Regularizer: None")
    benchmarks_reg_none = benchmark.run_benchmark(
        solver.lsr,
        matrix_representation=matrix_representation,
        lambd=lambd,
        regularizer=None,
    )

    print("Regularizer: smooth")
    benchmarks_reg_smooth = benchmark.run_benchmark(
        solver.lsr,
        matrix_representation=matrix_representation,
        lambd=lambd,
        regularizer="smooth",
    )

    print("Regularizer: abs")
    benchmarks_reg_abs = benchmark.run_benchmark(
        solver.lsr,
        matrix_representation=matrix_representation,
        lambd=lambd,
        regularizer="abs",
    )

    benchmark.save_benchmarks(
        [benchmarks_reg_none, benchmarks_reg_smooth, benchmarks_reg_abs],
        "least_squares_regularized",
    )


def bin_and_plot(arrays: list[np.ndarray], labels: list[str] = None, fname: str = "regularized_x_binned.png"):
    if labels is None:
        labels = [f"Array {i+1}" for i in range(len(arrays))]

    # flatten all values to compute common bin range
    all_values = np.concatenate(arrays)
    bin_min = np.floor(all_values.min() * 10) / 10
    bin_max = np.ceil(all_values.max() * 10) / 10
    bins = np.arange(bin_min, bin_max + 0.1, 0.1)
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]

    total_width = 0.8
    num_arrays = len(arrays)
    bar_width = total_width / num_arrays
    x = np.arange(len(bin_labels))

    plt.figure(figsize=(10, 6))

    for i, arr in enumerate(arrays):
        counts, _ = np.histogram(arr, bins=bins)
        plt.bar(x + i * bar_width, counts, width=bar_width, label=labels[i], edgecolor="black")

    plt.xticks(x + total_width / 2 - bar_width / 2, bin_labels, rotation=45, ha="right")
    plt.xlabel("Bins")
    plt.ylabel("Count")
    plt.title("X Coefficient Histogram")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"../../outputs/experiments/{fname}")
    plt.show()


def analyze():
    benchmarks = benchmark.load_benchmarks("least_squares_regularized")

    x_reg_none = benchmarks[0].x
    x_reg_smooth = benchmarks[1].x
    x_reg_abs = benchmarks[2].x

    bin_and_plot([x_reg_none, x_reg_smooth, x_reg_abs], labels=["None", "smooth", "abs"])

    benchmark.run_analysis(benchmarks, image, dirname="least_squares_regularized")


if __name__ == "__main__":
    stringart_directory: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory: Path = stringart_directory.parent.resolve()

    image_cropped = crop_image(image, crop_mode)
    solver = Solver(image_cropped, crop_mode, number_of_pegs=number_of_pegs, rasterization=rasterization)

    Benchmark.initialize_metadata(directory)
    benchmark = Benchmark(image, crop_mode, number_of_pegs, rasterization)

    if not os.path.exists("../../benchmarks/least_squares_regularized.json"):
        solve()

    analyze()
