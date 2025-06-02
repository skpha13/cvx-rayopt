import logging
import os
from pathlib import Path

from stringart.solver import Solver
from stringart.utils.image import ImageWrapper, crop_image
from stringart.utils.perf_analyzer import Benchmark
from stringart.utils.types import CropMode, Rasterization

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
number_of_pegs = 256
rasterization: Rasterization = "xiaolin-wu"
block_size: int = 8


def main():
    logging.basicConfig(level=logging.INFO)
    stringart_directory: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory: Path = stringart_directory.parent.resolve()

    image_cropped = crop_image(image, crop_mode)
    solver = Solver(
        image_cropped, crop_mode, number_of_pegs=number_of_pegs, rasterization=rasterization, block_size=block_size
    )

    Benchmark.initialize_metadata(directory)
    benchmark = Benchmark(image, crop_mode, number_of_pegs, rasterization, block_size=block_size)
    radon_results = benchmark.run_benchmark(
        solver.radon,
        uds=True,
        patience=10,
    )

    benchmark_results = [radon_results]
    benchmark.save_benchmarks(benchmark_results, "lena")
    benchmark.run_analysis(benchmark_results, 1 - image, "lena")


if __name__ == "__main__":
    main()
