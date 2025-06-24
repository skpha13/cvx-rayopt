import os
from pathlib import Path

from stringart.solver import Solver
from stringart.utils.image import ImageWrapper, crop_image
from stringart.utils.perf_analyzer import Benchmark, BenchmarkResult
from stringart.utils.types import CropMode, Rasterization
from tqdm import tqdm

crop_mode: CropMode = "center"
number_of_pegs = 256
rasterization: Rasterization = "xiaolin-wu"
block_size: int = 8

image_dir = Path("../../imgs/demo")
image_paths = sorted([file for file in image_dir.iterdir() if file.is_file()])
images = [
    ImageWrapper.apply_alpha_map_bw_to_rgba(
        crop_image(ImageWrapper.read_bw(path, inverted=False), crop_mode),
        ImageWrapper.alpha_map(crop_image(ImageWrapper.read_bw(path, inverted=True), crop_mode), crop_mode),
    )
    for path in image_paths
]


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
    for path in tqdm(image_paths, desc="Processing images"):
        result = run_benchmark(path)
        results.append(result)

    return results


def main():
    stringart_directory: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory: Path = stringart_directory.parent.resolve()
    Benchmark.initialize_metadata(directory)

    results = run_benchmarks()
    Benchmark.save_benchmarks(results, "demo")


if __name__ == "__main__":
    main()
