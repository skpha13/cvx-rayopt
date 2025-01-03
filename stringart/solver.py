from typing import List

import numpy as np
from numpy.linalg import lstsq
from scipy.sparse.linalg import lsqr

from stringart.line_algorithms.bresenham import Bresenham
from stringart.line_algorithms.matrix import MatrixGenerator
from stringart.utils.circle import compute_pegs
from stringart.utils.image import ImageWrapper, crop_image, find_radius_and_center_point
from stringart.utils.types import Method, Mode, Point


class Solver:
    """A class to compute string art procedurally using multiple methods.

    Parameters
    ----------
    image : ImageWrapper
        An object representing the image to be processed.
    image_mode : Mode
        The mode in which the image is being processed. This determines cropping behaviour.
    """

    # TODO: add docstring number_of_pegs

    def __init__(self, image: ImageWrapper, image_mode: Mode, number_of_pegs: int = 100):
        self.shape: tuple[int, ...] = image.get_shape()
        self.b: np.ndarray = image.flatten_image()
        self.image_mode: Mode = image_mode
        self.number_of_pegs: int = number_of_pegs

    def compute_solution(self, A: np.ndarray, x: np.ndarray) -> np.ndarray:
        solution = A @ x
        solution = np.clip(np.reshape(solution, shape=self.shape), a_min=0, a_max=1)
        solution = np.multiply(solution, 255).astype(np.uint8)
        solution = crop_image(solution, self.image_mode)

        return solution

    def least_squares(self, method: Method = "sparse") -> np.ndarray:
        """Solve the string art problem using the least squares method.

        Parameters
        ----------
        method : {'dense', 'sparse'}, optional
            The method to use for solving the least squares problem. Defaults to 'sparse'.
            - 'dense': Uses dense matrix operations via `numpy.linalg.lstsq`.
            - 'sparse': Uses sparse matrix operations via `scipy.sparse.linalg.lsqr`.

        Returns
        -------
        numpy.ndarray
            The reconstructed image as a 2D greyscale array.
            The pixel values are scaled between 0 and 255 and cropped according to the image mode.
        """

        A, pegs = MatrixGenerator.compute_matrix(self.shape, self.number_of_pegs, self.image_mode, method)

        x = None
        if method == "dense":
            x, _, _, _ = lstsq(A, self.b)
        elif method == "sparse":
            x = lsqr(A, self.b)[0]

        return self.compute_solution(A, x)

    def greedy(self, number_of_lines: int) -> np.ndarray:
        radius, center_point = find_radius_and_center_point(self.shape, self.image_mode)
        pegs: List[Point] = compute_pegs(
            number_of_pegs=self.number_of_pegs,
            radius=radius,
            center_point=center_point,
        )

        candidate_lines = MatrixGenerator.generate_dense_matrix(self.shape, pegs)
        rows, cols = candidate_lines.shape
        best_lines: set[int] = set()

        A = np.empty((rows, 0))
        x = None

        for step in range(number_of_lines):
            best_index = -1
            best_residual = np.inf

            for column_index in range(cols):
                # check if we already included the line
                if column_index in best_lines:
                    continue

                A_trial = np.column_stack((A, candidate_lines[:, column_index]))
                x_trial, _, _, _ = np.linalg.lstsq(A_trial, self.b)

                residual = self.b - A_trial @ x_trial
                residual = np.sum(np.square(residual))

                if residual < best_residual:
                    best_index = column_index
                    best_residual = residual

            if best_index == -1:
                raise RuntimeError(f"No index found at step: {step}.")

            A = np.column_stack((A, candidate_lines[:, best_index]))
            best_lines.add(best_index)
            x, _, _, _ = np.linalg.lstsq(A, self.b)

        return self.compute_solution(A, x)
