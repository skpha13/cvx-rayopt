from typing import List, cast

import numpy as np
import scipy
from numpy.linalg import lstsq
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr

from stringart.line_algorithms.matrix import MatrixGenerator
from stringart.utils.circle import compute_pegs
from stringart.utils.greedy_selector import GreedySelector
from stringart.utils.image import ImageWrapper, crop_image, find_radius_and_center_point
from stringart.utils.matching_pursuit import Greedy, MatchingPursuit, Orthogonal
from stringart.utils.types import CropMode, MatchingPursuitMethod, MatrixRepresentation, Point, Rasterization


class Solver:
    """A class to compute string art procedurally using multiple methods.

    Parameters
    ----------
    image : np.ndarray
        A numpy array representing the image to be processed.
    image_mode : CropMode
        The mode in which the image is being processed. This determines cropping behaviour.
    number_of_pegs : int, optional
        The number of pegs to be used in the string art computation. Default is 100.
    rasterization: Rasterization, optional
        If "xiaolin-wu", the line is generated using a rasterized algorithm (Xiaolin Wu's algorithm).
        If "bresenham", the line is generated using a non-rasterized algorithm (Bresenham's algorithm).
    """

    def __init__(
        self,
        image: np.ndarray,
        image_mode: CropMode | None = "center",
        number_of_pegs: int | None = 100,
        rasterization: Rasterization | None = "bresenham",
    ):
        image_mode: CropMode = image_mode if image_mode else "center"
        number_of_pegs = number_of_pegs if number_of_pegs else 100
        rasterization: Rasterization = rasterization if rasterization else "bresenham"

        self.shape: tuple[int, ...] = image.shape
        self.b: np.ndarray = ImageWrapper.histogram_equalization(image)  # preprocess image
        self.b = ImageWrapper.flatten_image(image).astype(np.float64)
        self.image_mode: CropMode = image_mode
        self.number_of_pegs: int = number_of_pegs
        self.rasterization: Rasterization = rasterization

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

    # TODO: think if we should expose binary to CLI
    def compute_solution_top_k(self, A: np.ndarray, x: np.ndarray, k: int = 1000, binary: bool = False) -> np.ndarray:
        """Computes the solution image using only the top-k elements from the input vector.

        Parameters
        ----------
        A : np.ndarray
            The transformation matrix used to generate the solution.
        x : np.ndarray
            The input vector representing the parameters of the transformation.
        k : int, optional
            The number of top elements to retain from the input vector, by default 1000.
        binary : bool, optional
            If True, converts the top-k vector to binary values (0 or 1). If False, retains original values
            for the top-k elements and sets others to 0.

        Returns
        -------
        np.ndarray
            The processed solution image. The image is clipped to values between
            0 and 1, scaled to 255, converted to an 8-bit format, and cropped
            according to the specified image mode.
        """

        value = x[np.argsort(x)[-k]]

        xp = x.copy()
        xp[xp < value] = 0
        # used to fully transform values to 0/1
        # if false it will keep the Least Squares coefficients
        if binary:
            xp[xp >= value] = 1
        xp = np.clip(xp, a_min=0, a_max=1)

        solution = A @ xp
        solution = np.clip(np.reshape(solution, shape=self.shape), a_min=0, a_max=1)
        solution = 1 - solution
        solution = np.multiply(solution, 255).astype(np.uint8)
        solution = crop_image(solution, self.image_mode)

        return solution

    def least_squares(
        self, matrix_representation: MatrixRepresentation | None = "sparse"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the string art problem using the least squares method.

        Parameters
        ----------
        matrix_representation : {'dense', 'sparse'}, optional
            The method to use for solving the least squares problem. Defaults to 'sparse'.
            - 'dense': Uses dense matrix operations via `numpy.linalg.lstsq`.
            - 'sparse': Uses sparse matrix operations via `scipy.sparse.linalg.lsqr`.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The initial matrix of column vectors representing lines to be drawn.
            And the x solution of the system.
        """
        matrix_representation: MatrixRepresentation = matrix_representation if matrix_representation else "sparse"

        A, pegs = MatrixGenerator.compute_matrix(
            self.shape, self.number_of_pegs, self.image_mode, matrix_representation, self.rasterization
        )

        x = None
        if matrix_representation == "dense":
            x, _, _, _ = lstsq(A, self.b)
        elif matrix_representation == "sparse":
            x = lsqr(A, self.b)[0]

        return A, x

    # TODO: think if we should let the user choose custom bounds
    def linear_least_squares(
        self, matrix_representation: MatrixRepresentation | None = "sparse", bounds: scipy.optimize.Bounds = (0, np.inf)
    ):
        """Solve the string art problem using the linear least squares method bounded to have positive x values.

        Parameters
        ----------
        matrix_representation : {'dense', 'sparse'}, optional
            The method to use for solving the least squares problem. Defaults to 'sparse'.
            - 'dense': Uses dense matrix operations via `numpy.linalg.lstsq`.
            - 'sparse': Uses sparse matrix operations via `scipy.sparse.linalg.lsqr`.

        bounds : tuple or scipy.optimize.Bounds, optional
            Lower and upper bounds on the solution. Defaults to non-negative values (0, np.inf).

        Returns
        -------
        A : ndarray
            The initial matrix of column vectors representing lines to be drawn.

        x : ndarray
            The solution vector `x` that minimizes the least squares problem `||Ax - b||^2`
            subject to the specified bounds.
        """

        matrix_representation: MatrixRepresentation = matrix_representation if matrix_representation else "sparse"

        A, _ = MatrixGenerator.compute_matrix(
            self.shape, self.number_of_pegs, self.image_mode, matrix_representation, self.rasterization
        )
        optimize_results: scipy.optimize.OptimizeResult = scipy.optimize.lsq_linear(A, self.b, bounds=bounds)

        return A, optimize_results.x

    def matching_pursuit(
        self,
        number_of_lines: int,
        mp_method: MatchingPursuitMethod | None = "orthogonal",
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Performs a matching pursuit algorithm to select the best lines from the candidate lines matrix,
        iteratively adding the top-k candidates to minimize the residual error with respect to
        the target vector `b`.

        Parameters
        ----------
        number_of_lines : int
           The number of lines to select and add to the solution.

        mp_method : MatchingPursuitMethod, optional
           The matching pursuit method, either "orthogonal" or "greedy". Default is "orthogonal".

        **kwargs:
            Additional parameters for the 'greedy' method, such as selector_type.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The initial matrix of column vectors representing lines to be drawn.
            And the x solution of the system.

        Notes
        -----
        The algorithm will stop early if no improvement is found in the residual between steps.
        It relies on a sparse matrix of candidate lines and uses the `lsqr` solver to solve the
        least squares problem at each step.

        The greedy approach iterates through the candidate lines, selecting the best candidates
        based on their dot product with the target vector `b` (or randomly, depending on the
        selector type) and minimizes the residual error in the least squares problem.
        """

        mp_method = mp_method if mp_method else "orthogonal"

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

        mp_instance: MatchingPursuit | None = None
        if mp_method == "greedy":
            selector_type = cast(GreedySelector, kwargs.get("selector_type", "dot-product"))
            mp_instance = Greedy(A, self.b, selector_type=selector_type)
        elif mp_method == "orthogonal":
            mp_instance = Orthogonal(A, self.b)  # Orthogonal doesn't use A matrix

        for step in range(number_of_lines):
            # exclude already selected lines
            remaining_lines_indices = all_line_indices - selected_lines
            remaining_candidate_lines = candidate_lines[:, list(remaining_lines_indices)]

            best_index = 0
            try:
                best_index = mp_instance.compute_best_column(remaining_candidate_lines)
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

        return A, x
