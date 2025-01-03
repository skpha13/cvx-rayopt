import os
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

    solver = Solver(image, mode, number_of_pegs=20)
    solution = solver.greedy(number_of_lines=100, selector_type="random")

    io.imsave(metadata.path / "outputs/lena_stringart_greedy.png", solution)
    io.imshow(solution)
    plt.show()


if __name__ == "__main__":
    main()
