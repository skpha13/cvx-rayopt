from typing import List

import numpy as np
from numpy.linalg import lstsq
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr

from stringart.line_algorithms.matrix import MatrixGenerator
from stringart.utils.circle import compute_pegs
from stringart.utils.greedy_selector import GreedySelector
from stringart.utils.image import ImageWrapper, crop_image, find_radius_and_center_point
from stringart.utils.matching_pursuit import Greedy, Orthogonal
from stringart.utils.types import Method, Mode, Point


class Solver:
    """A class to compute string art procedurally using multiple methods.

    Parameters
    ----------
    image : np.ndarray
        A numpy array representing the image to be processed.
    image_mode : Mode
        The mode in which the image is being processed. This determines cropping behaviour.
    number_of_pegs : int, optional
        The number of pegs to be used in the string art computation. Default is 100.
    """

    def __init__(self, image: np.ndarray, image_mode: Mode, number_of_pegs: int = 100):
        self.shape: tuple[int, ...] = image.shape
        self.b: np.ndarray = ImageWrapper.flatten_image(image)
        self.image_mode: Mode = image_mode
        self.number_of_pegs: int = number_of_pegs

    def compute_solution(self, A: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Computes the solution image for the string art procedure.

        Parameters
        ----------
        A : np.ndarray
           The transformation matrix used to generate the solution.
        x : np.ndarray
           The input vector representing the parameters of the transformation.

        Returns
        -------
        np.ndarray
           The processed solution image. The image is clipped to values between
           0 and 1, scaled to 255, converted to an 8-bit format, and cropped
           according to the specified image mode.
        """

        solution = A @ x
        solution = np.clip(np.reshape(solution, shape=self.shape), a_min=0, a_max=1)
        solution = 1 - solution
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

    def matching_pursuit(self, number_of_lines: int, selector_type: GreedySelector = "random") -> np.ndarray:
        """Performs a matching pursuit algorithm to select the best lines from the candidate lines matrix,
        iteratively adding the top-k candidates to minimize the residual error with respect to
        the target vector `b`.

        Parameters
        ----------
        number_of_lines : int
           The number of lines to select and add to the solution.

        selector_type : GreedySelector, optional
           The type of selector to use for candidate selection.
           Can be either "random" or "dot-product". Default is "random".

        Returns
        -------
        np.ndarray
           The final solution vector `x` after the greedy line selection process.

        Notes
        -----
        The algorithm will stop early if no improvement is found in the residual between steps.
        It relies on a sparse matrix of candidate lines and uses the `lsqr` solver to solve the
        least squares problem at each step.

        The greedy approach iterates through the candidate lines, selecting the best candidates
        based on their dot product with the target vector `b` (or randomly, depending on the
        selector type) and minimizes the residual error in the least squares problem.
        """

        radius, center_point = find_radius_and_center_point(self.shape, self.image_mode)
        pegs: List[Point] = compute_pegs(
            number_of_pegs=self.number_of_pegs,
            radius=radius,
            center_point=center_point,
        )

        candidate_lines = MatrixGenerator.generate_sparse_matrix(self.shape, pegs)
        rows, cols = candidate_lines.shape
        selected_lines: set[int] = set()
        all_line_indices = set(range(cols))
        past_residual = np.inf

        A = csr_matrix((rows, 0))
        x = None

        for step in range(number_of_lines):
            # exclude already selected lines
            remaining_lines_indices = all_line_indices - selected_lines
            remaining_candidate_lines = candidate_lines[:, list(remaining_lines_indices)]

            greedy = Greedy(A, self.b, selector_type=selector_type)
            omp = Orthogonal(A, self.b)
            try:
                # best_index = greedy.compute_best_column(remaining_candidate_lines)
                best_index = omp.compute_best_column(remaining_candidate_lines)
                best_index = list(remaining_lines_indices)[best_index]
            except ValueError:
                # it means no line was found
                break

            selected_lines.add(best_index)
            best_column = candidate_lines[:, best_index]

            A = hstack([A, best_column])
            x = lsqr(A, self.b)[0]

            # if the error does not decrease
            residual = np.linalg.norm(self.b - A @ x)
            if not residual <= past_residual:
                break

        return self.compute_solution(A, x)
