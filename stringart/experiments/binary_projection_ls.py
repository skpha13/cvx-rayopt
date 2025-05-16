import matplotlib.pyplot as plt
from stringart.solver import Solver
from stringart.utils.image import ImageWrapper, crop_image
from stringart.utils.types import CropMode

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"

# TODO: benchmark both of these + comparative analysis


def main():
    image_cropped = crop_image(image, crop_mode)
    solver = Solver(image_cropped, crop_mode, number_of_pegs=100)
    A, x = solver.binary_projection_ls("cvxopt", "sparse", max_iterations=3)
    solution = solver.compute_solution(A, x)

    plt.imshow(solution, cmap="grey")
    plt.show()
    plt.imsave("../../outputs/misc/binary_projection_ls.png", solution, cmap="grey")


if __name__ == "__main__":
    main()
