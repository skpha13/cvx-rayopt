import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np
from stringart.cli_functions import Configuration
from stringart.utils.image import ImageWrapper
from stringart.utils.types import CropMode, MatrixRepresentation, Metadata, Rasterization, SolverType

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
number_of_pegs = 10
matrix_representation: MatrixRepresentation = "sparse"
k = 500


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
        )


def run_config(config: Configuration):
    return config.run_config_lite()


def run_configs(configs: List[Configuration]) -> np.ndarray:
    results = [[], []]

    with ProcessPoolExecutor() as executor:
        future_to_config = {executor.submit(run_config, config): config for config in configs}

        for future in as_completed(future_to_config):
            config = future_to_config[future]

            result = future.result()

            row_index = 0 if config.rasterization == "bresenham" else 1
            results[row_index].append(result)

    return np.array(results)


def run_rendering_experiments():
    pass


def analyze_experiments():
    pass


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
    print(results.shape)


if __name__ == "__main__":
    main()
