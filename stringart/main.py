import os
import time
from pathlib import Path

from matplotlib import pyplot as plt
from skimage import io

from stringart.solver import Solver
from stringart.utils.image import ImageWrapper
from stringart.utils.types import Metadata, Mode


def main() -> None:
    stringart_directory: Path = Path(os.path.dirname(os.path.abspath(__file__)))
    directory: Path = stringart_directory.parent.resolve()

    metadata = Metadata(directory)

    image = ImageWrapper()
    image.read_bw(metadata.path / "imgs/lena.png")
    mode: Mode = "center"

    # solver = Solver(image, mode, number_of_pegs=100)
    # solution = solver.least_squares("sparse")

    time_start = time.time()

    solver = Solver(image, mode, number_of_pegs=50)
    solution = solver.greedy(number_of_lines=1000, selector_type="random")

    time_end = time.time()
    elapsed_time = time_end - time_start

    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    milliseconds = int((elapsed_time % 1) * 1000)

    print(f"Elapsed time: {minutes:02d}:{seconds:02d}.{milliseconds:03d}")

    io.imsave(metadata.path / "outputs/lena_stringart_greedy.png", solution)
    io.imshow(solution)
    plt.show()


if __name__ == "__main__":
    main()
