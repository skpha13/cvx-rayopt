import matplotlib.pyplot as plt
import numpy as np
from stringart.optimize.downsampling import UDSLoss
from stringart.solver import Solver
from stringart.utils.circle import compute_pegs
from stringart.utils.image import ImageWrapper, crop_image, find_radius_and_center_point
from stringart.utils.types import CropMode, MatrixRepresentation, Point, Rasterization

image_path = "../../imgs/lena.png"
image = ImageWrapper.read_bw(image_path)
shape = image.shape
crop_mode: CropMode = "center"
matrix_representation: MatrixRepresentation = "sparse"
rasterization: Rasterization = "xiaolin-wu"
number_of_pegs: list[int] = [32, 64, 128, 256]
number_of_lines: int = 1000
block_sizes: list[int] = [2, 4, 8, 16]


def udps(x: np.ndarray, number_of_pegs: int, block_size: int = 2) -> tuple[np.floating, np.ndarray]:
    residual_fn = UDSLoss(image, crop_mode, number_of_pegs, rasterization, block_size=block_size)
    return residual_fn(x)


def solve_lls(number_of_pegs: int):
    image_cropped = crop_image(image, crop_mode)
    solver = Solver(image_cropped, crop_mode, number_of_pegs=number_of_pegs, rasterization=rasterization)
    A, x, residuals = solver.lls("sparse", bounds=(0, 1))

    k = min(number_of_lines, len(x))
    value = x[np.argsort(x)[-k]]

    xp = x.copy()
    xp[xp < value] = 0
    xp[xp >= value] = 1

    residuals = []
    solutions = []
    for block_size in block_sizes:
        residual, solution = udps(xp, number_of_pegs, block_size)
        residuals.append(residual)
        solutions.append(solution)

    return solutions, residuals


def compare_varying_pegs_lls():
    solutions = []
    residuals = []
    manhattan_distances = []
    best_residuals = []
    best_block_sizes = []

    for n in number_of_pegs:
        block_size_solutions, block_size_residuals = solve_lls(n)
        solutions.append(block_size_solutions)
        residuals.append(block_size_residuals)

        # compute pegs and manhattan distance between first two pegs
        radius, center_point = find_radius_and_center_point(shape, crop_mode)
        pegs: list[Point] = compute_pegs(
            number_of_pegs=n,
            radius=radius,
            center_point=center_point,
        )

        pointA = pegs[0]
        pointB = pegs[1]
        manhattan_distance = abs(pointA.x - pointB.x) + abs(pointA.y - pointB.y)
        manhattan_distances.append(manhattan_distance)

        # find the best residual and corresponding block size
        best_idx = np.argmin(block_size_residuals)
        best_residual = block_size_residuals[best_idx]
        best_block_size = block_sizes[best_idx]
        best_residuals.append(best_residual)
        best_block_sizes.append(best_block_size)

        print(
            f"Pegs: {n}, Manhattan Distance: {manhattan_distance}, "
            f"Best Residual: {best_residual:.4f}, Best Block Size: {best_block_size}"
        )

    # plot manhattan distance vs best residual
    plt.figure(figsize=(8, 5))
    plt.plot(manhattan_distances, best_residuals, marker="o")

    for i, n in enumerate(number_of_pegs):
        plt.annotate(f"N: {n}", (manhattan_distances[i], best_residuals[i]), fontsize=8)

    plt.xlabel("Manhattan Distance Between First Two Pegs")
    plt.ylabel("Best Residual")
    plt.title("Best Residual vs. Peg-to-Peg Manhattan Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../../outputs/experiments/varying_number_of_pegs/lls_residuals_by_manhattan_distance.png")
    plt.show()

    # plot images
    num_rows = len(number_of_pegs)
    num_cols = len(block_sizes)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axs = np.atleast_2d(axs)
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axs[i, j]
            ax.imshow(solutions[i][j], cmap="gray")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if i == 0:
                ax.set_title(f"Block Size: {block_sizes[j]}", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"N: {number_of_pegs[i]}", rotation=0, ha="right", va="center", fontsize=12)
                ax.yaxis.set_label_coords(-0.2, 0.5)

    plt.tight_layout()
    plt.savefig(f"../../outputs/experiments/varying_number_of_pegs/lls_images_by_pegs_and_block_size.png")
    plt.show()

    # plot residuals
    plt.figure(figsize=(8, 5))
    for i, n in enumerate(number_of_pegs):
        plt.plot(block_sizes, residuals[i], marker="o", label=f"N: {n}")

    plt.xlabel("Block Size")
    plt.xticks(block_sizes)
    plt.ylabel("Residual")
    plt.title("Residuals vs. Block Size for Different Peg Counts")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../../outputs/experiments/varying_number_of_pegs/lls_residuals_by_pegs_and_block_size.png")
    plt.show()

    column_index = 2
    num_images = len(number_of_pegs)
    fig, axs = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
    axs = np.atleast_1d(axs)

    for i in range(num_images):
        ax = axs[i]
        ax.imshow(solutions[i][column_index], cmap="gray")
        ax.axis("off")
        ax.set_title(f"N: {number_of_pegs[i]}")

    plt.tight_layout()
    plt.savefig(f"../../outputs/experiments/varying_number_of_pegs/lls_column_{column_index}_as_row.png")
    plt.show()


def solve_ls(number_of_pegs: int):
    image_cropped = crop_image(image, crop_mode)
    solver = Solver(image_cropped, crop_mode, number_of_pegs=number_of_pegs)
    A, x, residuals = solver.ls("sparse")
    solution = solver.compute_solution(A, x)

    return solution, residuals[-1]


def compare_varying_pegs_ls():
    solutions = []
    residuals = []
    for n in number_of_pegs:
        solution, residual = solve_ls(n)
        solutions.append(solution)
        residuals.append(residual)

    # plot images
    fig, axs = plt.subplots(1, len(solutions), figsize=(4 * len(solutions), 4))
    for i, img in enumerate(solutions):
        axs[i].imshow(img, cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(f"N: {number_of_pegs[i]}")

    plt.suptitle("Least Squares for Varying Pegs")
    plt.tight_layout()
    plt.savefig(f"../../outputs/experiments/varying_number_of_pegs/ls_images.png")
    plt.show()

    # plot residuals
    plt.figure(figsize=(6, 4))
    plt.plot(number_of_pegs, residuals, marker="o")
    plt.xticks(number_of_pegs)
    plt.xlabel("Number of Pegs")
    plt.ylabel("Residual")
    plt.title("Residuals vs. Number of Pegs")
    plt.grid(True)
    plt.savefig(f"../../outputs/experiments/varying_number_of_pegs/ls_residuals.png")
    plt.show()


def main():
    compare_varying_pegs_ls()
    compare_varying_pegs_lls()


if __name__ == "__main__":
    main()
