from matplotlib import pyplot as plt
from stringart.misc.least_squares_rounding_trial import compute_solution
from stringart.misc.linear_least_squares import linear_least_squares
from stringart.utils.image import ImageWrapper
from stringart.utils.types import CropMode, MatrixRepresentation, Rasterization

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
number_of_pegs = 100
matrix_representation: MatrixRepresentation = "sparse"
rasterization: Rasterization = "xiaolin-wu"


def main():
    src = ImageWrapper.histogram_equalization(image)
    A, x = linear_least_squares(
        src,
        shape,
        number_of_pegs,
        crop_mode,
        matrix_representation,
        rasterization=rasterization,
    )
    solution = compute_solution(A, x)

    plt.imshow(solution, cmap="grey")
    plt.axis("off")
    plt.show()
    plt.imsave("../../outputs/misc/preprocess_xiaolin_wu_lena.png", solution, cmap="grey")


if __name__ == "__main__":
    main()
