import os
from pathlib import Path

import imageio
import numpy as np
from stringart.optimize.downsampling import UDSLoss
from stringart.utils.image import ImageWrapper
from stringart.utils.perf_analyzer import Benchmark, BenchmarkResult
from tqdm import tqdm


def animation(x: np.ndarray, loss: UDSLoss, filename: str = "animation.mp4"):
    x_temp = np.zeros(x.shape)
    indices = np.flatnonzero(x)
    frames = []

    for i in tqdm(indices, desc="Drawing lines"):
        x_temp[i] = 1
        _, image = loss(x_temp)

        # rgb dimension conversion from grayscale
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        frames.append(image)

    fps = 30
    with imageio.get_writer(filename, format="ffmpeg", mode="I", fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)


def generate_animations(xs: list[np.ndarray], loss: UDSLoss):
    for i, x in enumerate(xs):
        animation(x, loss, filename=f"../../outputs/demo/animation_{i}.mp4")


if __name__ == "__main__":
    stringart_directory: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    directory: Path = stringart_directory.parent.resolve()
    Benchmark.initialize_metadata(directory)

    results: list[BenchmarkResult] = Benchmark.load_benchmarks("radon_quantitative")
    image = ImageWrapper.scale_image(results[0].output_image)
    number_of_pegs = results[0].number_of_pegs
    crop_mode = results[0].crop_mode
    rasterization = results[0].rasterization
    block_size = results[0].block_size

    xs = [np.array(results[0].x), np.array(results[2].x), np.array(results[8].x)]
    loss = UDSLoss(image, crop_mode, number_of_pegs, rasterization, block_size)

    generate_animations(xs, loss)
