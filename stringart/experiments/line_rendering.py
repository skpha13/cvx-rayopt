import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import normalized_root_mse
from stringart.cli_functions import Configuration
from stringart.utils.image import ImageWrapper, crop_image
from stringart.utils.perf_analyzer import prepare_diff_images
from stringart.utils.types import CropMode, MatrixRepresentation, Metadata, Rasterization, SolverType

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
number_of_pegs = 100
matrix_representation: MatrixRepresentation = "sparse"
k = 500
fontsize = 30


class ConfigurationFactory:
    def __init__(
        self,
        metadata: Metadata,
        image_path,
        number_of_pegs: int,
        crop_mode: CropMode,
        matrix_representation: MatrixRepresentation,
    ):
        self.metadata = metadata
        self.image_path = image_path
        self.number_of_pegs = number_of_pegs
        self.crop_mode = crop_mode
        self.matrix_representation = matrix_representation

    def create(
        self,
        solver: SolverType = "least-squares",
        rasterization: Rasterization = "bresenham",
        number_of_lines: int | None = None,
        binary: bool = False,
    ):
        return Configuration(
            metadata=self.metadata,
            command="solve",
            solver=solver,
            image_path=self.image_path,
            number_of_pegs=self.number_of_pegs,
            crop_mode=self.crop_mode,
            rasterization=rasterization,
            matrix_representation=self.matrix_representation,
            mp_method=None,
            number_of_lines=number_of_lines,
            selector_type=None,
            binary=binary,
            qp_solver=None,
            k=None,
            max_iterations=None,
        )


def run_config(config: Configuration, index: int):
    return index, config.run_config_lite()


def run_configs(configs: List[Configuration]) -> np.ndarray:
    nconf = len(configs) // 2
    results = [[None for _ in range(nconf)], [None for _ in range(nconf)]]

    with ProcessPoolExecutor() as executor:
        future_to_config = {executor.submit(run_config, config, index): config for index, config in enumerate(configs)}

        for future in as_completed(future_to_config):
            config = future_to_config[future]

            index, result = future.result()

            row_index = 0 if config.rasterization == "bresenham" else 1
            results[row_index][index % nconf] = result

    return np.array(results)


def plot_row(axs: Any, imgs: np.ndarray, labels: List[str] | None, cmap="gray") -> None:
    for index, ax in enumerate(axs):
        ax.imshow(imgs[index], cmap=cmap)

        if labels is not None:
            ax.text(
                0.5,
                -0.1,
                f"{labels[index]}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=fontsize,
            )


def analyze_results(src: np.ndarray, results: np.ndarray, cmap="gray", cmap_diff="inferno") -> None:
    src = crop_image(src, crop_mode)
    diff_images = ImageWrapper.scale_image(np.abs(src - ImageWrapper.scale_image(results)))
    diff_images = np.array(
        [
            prepare_diff_images(list(diff_images[0]), crop_mode, cmap_diff),
            prepare_diff_images(list(diff_images[1]), crop_mode, cmap_diff),
        ]
    )
    rmses = [
        [normalized_root_mse(src, ImageWrapper.scale_image(test_image)) for test_image in results[0]],
        [normalized_root_mse(src, ImageWrapper.scale_image(test_image)) for test_image in results[1]],
    ]
    src = ImageWrapper.apply_alpha_map_bw_to_rgba(src, ImageWrapper.alpha_map(src, crop_mode))

    fig, axs = plt.subplots(5, 6, figsize=(32, 28.8), constrained_layout=True)

    for row in axs:
        for ax in row:
            ax.axis("off")

    axs[0, 0].imshow(src, cmap=cmap)
    axs[0, 0].text(
        0.5,
        -0.1,
        f"Target Image",
        ha="center",
        va="center",
        transform=axs[0, 0].transAxes,
        fontsize=fontsize,
    )

    def get_rms(row, column):
        return f"{rmses[row][column]:.4f}"

    plot_row(axs[1], results[0], labels=None, cmap=cmap)
    plot_row(
        axs[2],
        diff_images[0],
        labels=[
            f"(a) RMS = {get_rms(0, 0)}",
            f"(b) RMS = {get_rms(0, 1)}",
            f"(c) RMS = {get_rms(0, 2)}",
            f"(d) RMS = {get_rms(0, 3)}",
            f"(e) RMS = {get_rms(0, 4)}",
            f"(f) RMS = {get_rms(0, 5)}",
        ],
        cmap=cmap_diff,
    )

    plot_row(axs[3], results[1], labels=None, cmap=cmap)
    plot_row(
        axs[4],
        diff_images[1],
        labels=[
            f"(g) RMS = {get_rms(1, 0)}",
            f"(h) RMS = {get_rms(1, 1)}",
            f"(i) RMS = {get_rms(1, 2)}",
            f"(j) RMS = {get_rms(1, 3)}",
            f"(k) RMS = {get_rms(1, 4)}",
            f"(l) RMS = {get_rms(1, 5)}",
        ],
        cmap=cmap_diff,
    )

    fig.show()
    fig.savefig("../../outputs/experiments/line_rendering_comparative_analysis.png")


def main():
    stringart_directory: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory: Path = stringart_directory.parent.resolve()
    metadata = Metadata(directory)

    config_factory = ConfigurationFactory(metadata, image_path, number_of_pegs, crop_mode, matrix_representation)

    # fmt: off
    configs = [
        config_factory.create("least-squares", "bresenham", None),
        config_factory.create("least-squares", "bresenham", k),
        config_factory.create("least-squares", "bresenham", k, binary=True),
        config_factory.create("linear-least-squares", "bresenham", None),
        config_factory.create("linear-least-squares", "bresenham", k),
        config_factory.create("linear-least-squares", "bresenham", k, binary=True),

        config_factory.create("least-squares", "xiaolin-wu", None),
        config_factory.create("least-squares", "xiaolin-wu", k),
        config_factory.create("least-squares", "xiaolin-wu", k, binary=True),
        config_factory.create("linear-least-squares", "xiaolin-wu", None),
        config_factory.create("linear-least-squares", "xiaolin-wu", k),
        config_factory.create("linear-least-squares", "xiaolin-wu", k, binary=True),
    ]
    # fmt: on

    results = run_configs(configs)
    analyze_results(1 - image, results)  # invert image back to normal bw


if __name__ == "__main__":
    main()
