import logging
from typing import List, Tuple, cast

import numpy as np
import scipy
from cvxopt import matrix, solvers
from numpy.linalg import lstsq
from scipy.sparse import csc_matrix, hstack
from scipy.sparse.linalg import lsqr
from tqdm import tqdm

from stringart.line.matrix import MatrixGenerator
from stringart.mp.greedy_selector import GreedySelector
from stringart.mp.matching_pursuit import Greedy, MatchingPursuit, Orthogonal
from stringart.optimize.downsampling import UDSLoss
from stringart.optimize.regularization import (
    AbsoluteValueRegularizer,
    NoRegularizer,
    Regularizer,
    SmoothRegularizer,
    WeightedRegularizer,
)
from stringart.utils.circle import compute_line_lengths, compute_pegs
from stringart.utils.image import ImageWrapper, crop_image, find_radius_and_center_point
from stringart.utils.types import (
    CropMode,
    MatchingPursuitMethod,
    MatrixRepresentation,
    Point,
    QPSolvers,
    Rasterization,
    RegularizationType,
)

logger = logging.getLogger(__name__)


class Solver:
    """A class to compute string art procedurally using multiple methods.

    Parameters
    ----------
    image : np.ndarray
        A numpy array representing the image to be processed.
    crop_mode : CropMode
        The mode in which the image is being processed. This determines cropping behaviour.
    number_of_pegs : int, optional
        The number of pegs to be used in the string art computation. Default is 128.
    rasterization: Rasterization, optional
        If "xiaolin-wu", the line is generated using a rasterized algorithm (Xiaolin Wu's algorithm).
        If "bresenham", the line is generated using a non-rasterized algorithm (Bresenham's algorithm).
    block_size : int
        Size of the square block used in downsampling via averaging.
    """

    def __init__(
        self,
        image: np.ndarray,
        crop_mode: CropMode | None = "center",
        number_of_pegs: int | None = 128,
        rasterization: Rasterization | None = "xiaolin-wu",
        block_size: int | None = None,
    ):
        crop_mode: CropMode = crop_mode if crop_mode else "center"
        number_of_pegs = number_of_pegs if number_of_pegs else 128
        rasterization: Rasterization = rasterization if rasterization else "xiaolin-wu"

        image = crop_image(image, crop_mode=crop_mode)

        self.shape: tuple[int, ...] = image.shape
        self.crop_mode: CropMode = crop_mode
        self.number_of_pegs: int = number_of_pegs
        self.rasterization: Rasterization = rasterization
        self.block_size = block_size

        self.b: np.ndarray = ImageWrapper.histogram_equalization(image)  # preprocess image
        self.b = ImageWrapper.flatten_image(self.b).astype(np.float64)

        self.residual_fn: UDSLoss | None = None
        if self.block_size is not None:
            self.residual_fn = UDSLoss(image, crop_mode, number_of_pegs, rasterization, block_size=self.block_size)

    def compute_solution(self, A: np.ndarray, x: np.ndarray, uds: bool = False) -> np.ndarray:
        """Computes the solution image for the string art procedure.

        Parameters
        ----------
        A : np.ndarray
            The transformation matrix used to generate the solution.
        x : np.ndarray
            The input vector representing the parameters of the transformation.
        uds: bool
            If set, it will run the UDS compute solution. Defaults to `False`.

        Returns
        -------
        np.ndarray
           The processed solution image. The image is clipped to values between
           0 and 1, scaled to 255, converted to an 8-bit format, and cropped
           according to the specified image mode.
        """

        if self.residual_fn is not None and uds == True:
            _, solution = self.residual_fn(x)
        else:
            solution = A @ x
            solution = np.clip(np.reshape(solution, self.shape), a_min=0, a_max=1)
            solution = 1 - solution
            solution = np.multiply(solution, 255).astype(np.uint8)

        return solution

    @classmethod
    def post_process_x(cls, x: np.ndarray, k: int = 1000, binary: bool = False) -> np.ndarray:
        """Post processes the `x` system solution vector using only the top-k elements from the input vector.

        Parameters
        ----------
        x : np.ndarray
            The input vector representing the system coefficients.
        k : int, optional
            The number of top elements to retain from the input vector, by default 1000.
        binary : bool, optional
            If True, converts the top-k vector to binary values (0 or 1). If False, retains original values
            for the top-k elements and sets others to 0.

        Returns
        -------
        np.ndarray
            The processed `x` coefficients. They are clipped to [0, 1].
        """

        k = min(k, len(x))
        value = x[np.argsort(x)[-k]]

        xp = x.copy()
        xp[xp < value] = 0
        # used to fully transform values to 0/1
        # if false it will keep the Least Squares coefficients
        if binary:
            xp[xp >= value] = 1
        xp = np.clip(xp, a_min=0, a_max=1)

        return xp

    @classmethod
    def get_number_of_lines(cls, k: int, shape: Tuple[int, ...]) -> int:
        return min(k, shape[1])

    def ls(
        self,
        matrix_representation: MatrixRepresentation | None = "sparse",
        number_of_lines: int | None = None,
        binary: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[np.floating]]:
        """Solve the string art problem using the least squares method.

        Parameters
        ----------
        matrix_representation : {'dense', 'sparse'}, optional
            The method to use for solving the least squares problem. Defaults to 'sparse'.
            - 'dense': Uses dense matrix operations via `numpy.linalg.lstsq`.
            - 'sparse': Uses sparse matrix operations via `scipy.sparse.linalg.lsqr`.
        number_of_lines : int | None
            The number of lines to select and add to the solution. Defaults to `None`.
        binary : bool, optional
            If True, converts the top-k vector to binary values (0 or 1). If False, retains original values
            for the top-k elements and sets others to 0.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[np.floating]]
            - The initial matrix of column vectors representing lines to be drawn.
            - The x solution of the system.
            - The residuals history.
        """
        matrix_representation: MatrixRepresentation = matrix_representation if matrix_representation else "sparse"
        logger.info(f"Least Squares: {matrix_representation}")

        A = MatrixGenerator.compute_matrix(
            self.shape, self.number_of_pegs, self.crop_mode, matrix_representation, self.rasterization
        )

        x = None
        if matrix_representation == "dense":
            x, _, _, _ = lstsq(A, self.b)
        elif matrix_representation == "sparse":
            x = lsqr(A, self.b)[0]

        if number_of_lines is not None:
            x = self.post_process_x(x, self.get_number_of_lines(number_of_lines, A.shape), binary)

        if self.residual_fn is not None and binary == True:
            residual, _ = self.residual_fn(x)
        else:
            residual = np.linalg.norm(self.b - A @ x)
        logger.info(f"Residual: {residual:.6f}")

        return A, x, [residual]

    def lls(
        self,
        matrix_representation: MatrixRepresentation | None = "sparse",
        number_of_lines: int | None = None,
        binary: bool | None = None,
        bounds: scipy.optimize.Bounds = (0, 1),
    ) -> tuple[np.ndarray, np.ndarray, list[np.floating]]:
        """Solve the string art problem using the linear least squares method bounded to have positive x values.

        Parameters
        ----------
        matrix_representation : {'dense', 'sparse'}, optional
            The method to use for solving the least squares problem. Defaults to 'sparse'.
            - 'dense': Uses dense matrix operations via `numpy.linalg.lstsq`.
            - 'sparse': Uses sparse matrix operations via `scipy.sparse.linalg.lsqr`.
        number_of_lines : int | None
            The number of lines to select and add to the solution. Defaults to `None`.
        binary : bool, optional
            If True, converts the top-k vector to binary values (0 or 1). If False, retains original values
            for the top-k elements and sets others to 0.
        bounds : tuple or scipy.optimize.Bounds, optional
            Lower and upper bounds on the solution. Defaults to non-negative values (0, 1).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[np.floating]]
            - The initial matrix of column vectors representing lines to be drawn.
            - The x solution of the system.
            - The residuals history.
        """
        matrix_representation: MatrixRepresentation = matrix_representation if matrix_representation else "sparse"
        logger.info(f"Linear Least Squares: {matrix_representation}")

        A = MatrixGenerator.compute_matrix(
            self.shape, self.number_of_pegs, self.crop_mode, matrix_representation, self.rasterization
        )
        optimize_results: scipy.optimize.OptimizeResult = scipy.optimize.lsq_linear(A, self.b, bounds=bounds)
        x = optimize_results.x

        if number_of_lines is not None:
            x = self.post_process_x(x, self.get_number_of_lines(number_of_lines, A.shape), binary)

        if self.residual_fn is not None and binary == True:
            residual, _ = self.residual_fn(x)
        else:
            residual = np.linalg.norm(self.b - A @ x)
        logger.info(f"Residual: {residual:.6f}")

        return A, x, [residual]

    def mp(
        self,
        number_of_lines: int,
        mp_method: MatchingPursuitMethod | None = "orthogonal",
        **kwargs,
    ) -> tuple[np.ndarray | csc_matrix, np.ndarray, list[np.floating]]:
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
        tuple[np.ndarray, np.ndarray, list[np.floating]]
            - The initial matrix of column vectors representing lines to be drawn.
            - The x solution of the system.
            - The residuals history.

        Notes
        -----
        The algorithm will stop early if no improvement is found in the residual between steps.
        It relies on a sparse matrix of candidate lines and uses the `lsqr` solver to solve the
        least squares problem at each step.

        The greedy approach iterates through the candidate lines, selecting the best candidates
        based on their dot product with the target vector `b` (or randomly, depending on the
        selector type) and minimizes the residual error in the least squares problem.
        """
        if number_of_lines is None:
            raise ValueError(
                "The 'number_of_lines' attribute is required when using the 'mp' solver and cannot be None."
            )

        mp_method = mp_method if mp_method else "orthogonal"
        logger.info(f"Matching Pursuit: {mp_method}")

        radius, center_point = find_radius_and_center_point(self.shape, self.crop_mode)
        pegs: List[Point] = compute_pegs(
            number_of_pegs=self.number_of_pegs,
            radius=radius,
            center_point=center_point,
        )

        candidate_lines = MatrixGenerator.generate_sparse_matrix(self.shape, pegs, self.rasterization)
        rows, cols = candidate_lines.shape
        number_of_lines = min(cols, number_of_lines)
        selected_lines: set[int] = set()
        all_line_indices = set(range(cols))
        past_residual = np.inf
        residual_history = []

        A = csc_matrix((rows, 0))
        x = None

        mp_instance: MatchingPursuit | None = None
        if mp_method == "greedy":
            selector_type = cast(GreedySelector, kwargs.get("selector_type", "dot-product"))
            mp_instance = Greedy(A, self.b, selector_type=selector_type, residual_fn=self.residual_fn)
        elif mp_method == "orthogonal":
            mp_instance = Orthogonal(A, self.b)  # Orthogonal doesn't use A matrix

        for step in tqdm(range(number_of_lines), desc="Selecting Lines"):
            logger.info(f"Step {step+1}/{number_of_lines}")
            logger.info("-" * 30)

            # exclude already selected lines
            remaining_lines_indices = all_line_indices - selected_lines
            remaining_candidate_lines = candidate_lines[:, list(remaining_lines_indices)]

            best_index = 0
            try:
                best_index = mp_instance.compute_best_column(
                    remaining_candidate_lines,
                    selected_lines=selected_lines,
                    remaining_lines_indices=remaining_lines_indices,
                )
                best_index = list(remaining_lines_indices)[best_index]
            except ValueError:
                # it means no line was found
                logger.info(
                    "No suitable line found by the matching pursuit method in the remaining candidates. Stopping selection early."
                )
                break

            selected_lines.add(best_index)
            best_column = candidate_lines[:, best_index]

            A = hstack([A, best_column])
            x = lsqr(A, self.b)[0]

            # if the error does not decrease
            residual, _ = self.residual_fn(x, x_indices=selected_lines)
            residual_history.append(residual)
            logger.info(f"Residual Check — Previous: {past_residual:.6f}, Current: {residual:.6f}")

            if not residual < past_residual:
                logger.info(f"Residual did not decrease (delta = {residual - past_residual:.6f}). Stopping early.")
                break
            past_residual = residual

        x_correct_shape = np.zeros(candidate_lines.shape[1])
        x_correct_shape[list(selected_lines)] = 1

        return candidate_lines, x_correct_shape, residual_history

    @classmethod
    def solve_qp_cvxopt(cls, A: np.ndarray, b: np.ndarray, regularizer: Regularizer | None = None, lambd: float = 0.1):
        """Solves a constrained quadratic program using CVXOPT.

        The optimization problem is formulated as:
            minimize (1/2)x^T P x + q^T x
            subject to 0 <= x <= 1

        Where:
            P = 2 * A^T A
            q = -2 * A^T b

        Parameters
        ----------
        A : np.ndarray
            The input matrix of shape (m, n). Can be a sparse matrix.
        b : np.ndarray
            The target vector of shape (m,).
        regularizer : Regularizer or None, optional
            An object that defines regularization behavior. If None, no regularization is applied
            (i.e., `NoRegularizer()` is used).
        lambd : float, optional
            Regularization strength. Only used if a `regularizer` is provided. Default is 0.1.

        Returns
        -------
        x : np.ndarray
            The solution vector of shape (n,), constrained to [0, 1].
        """
        if hasattr(A, "toarray"):
            A = A.toarray()

        solvers.options["abstol"] = 1e-8
        solvers.options["reltol"] = 1e-8
        solvers.options["feastol"] = 1e-8
        solvers.options["show_progress"] = False

        m, n = A.shape
        P = 2 * (A.T @ A)
        q = -2 * (A.T @ b)

        if regularizer is None:
            regularizer = NoRegularizer()

        P_reg, q_reg, G, h = regularizer.prepare_matrices(P, q, n, lambd)

        P_cvx = matrix(P_reg.astype(np.float64))
        q_cvx = matrix(q_reg.astype(np.float64))
        G_cvx = matrix(G.astype(np.float64))
        h_cvx = matrix(h.astype(np.float64))

        solution = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)

        return regularizer.post_process(solution, n)

    def bpls(
        self,
        solver: QPSolvers | None = "cvxopt",
        matrix_representation: MatrixRepresentation | None = "sparse",
        k: int | None = 3,
        max_iterations: int | None = 100,
        lambd: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[np.floating]]:
        """Projects the solution of a least squares problem to a binary space using iterative top-k selection.

        This method iteratively fixes `k` variables with the highest values from a constrained least squares
        solution to 1, while keeping others free, until all variables are fixed. The result is a binary vector
        approximating the original least squares solution.

        Parameters
        ----------
        solver : QPSolvers or None, default "cvxopt"
            Solver to use for the constrained least squares problem. Supported: "cvxopt" or "scipy" for "scipy.optimize.lsq_linear".
        matrix_representation : MatrixRepresentation or None, default="sparse"
            Format for matrix construction, e.g., "sparse" or "dense".
        k : int, default 10
            Number of variables to fix to 1 in each iteration.
        max_iterations : int, default 100
            Maximum number of iterations for binary projection.
        lambd: float, default None
             If provided it will apply a weighted entropy-like regularization using the formula x(1−x).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[np.floating]]
            - The initial matrix of column vectors representing lines to be drawn.
            - The binary x solution of the system, where entries are either 1 or 0.
            - The residuals history.
        """
        solver: QPSolvers = solver if solver else "cvxopt"
        logger.info(f"Binary Projection Least Squares: {solver}, lambda={lambd}")

        matrix_representation: MatrixRepresentation = matrix_representation if matrix_representation else "sparse"
        k = k if k else 3
        max_iterations = max_iterations if max_iterations else 100

        A = MatrixGenerator.compute_matrix(
            self.shape, self.number_of_pegs, self.crop_mode, matrix_representation, self.rasterization
        )

        n = A.shape[1]
        x_fixed = np.full(n, np.nan)  # nan means unfixed
        set1 = set()

        regularizer = WeightedRegularizer(n) if lambd is not None else None

        past_residual = np.inf
        residual_history = []
        for iteration in tqdm(range(max_iterations), desc="Iterating"):
            logger.info(f"Iteration {iteration+1}/{max_iterations}")
            logger.info("-" * 30)

            free_indices = np.isnan(x_fixed)

            # Step 1: solve least squares for current fixed values
            A_free = A[:, free_indices]
            b_adjusted = self.b.copy()

            if set1:
                A_fixed_1 = A[:, list(set1)]
                b_adjusted -= A_fixed_1 @ np.ones(len(set1))  # sum them up, then subtract

            bounds = (0, 1)
            if solver == "cvxopt":
                x_free = self.solve_qp_cvxopt(A_free, b_adjusted, regularizer, lambd)
            else:
                result = scipy.optimize.lsq_linear(A_free, b_adjusted, bounds=bounds)
                x_free = result.x

            # Step 2: find top-k variables to fix to 1
            free_idx_array = np.where(free_indices)[0]
            top_k = min(k, len(x_free))
            top_k_indices = np.argsort(-x_free)[:top_k]  # descending order
            chosen_indices = free_idx_array[top_k_indices]

            for idx in chosen_indices:
                x_fixed[idx] = 1
                set1.add(idx)

            # stop if all variables are fixed
            if np.all(~np.isnan(x_fixed)):
                break

            # if the error does not decrease
            x_residual = np.zeros(n)
            x_residual[~np.isnan(x_fixed)] = x_fixed[~np.isnan(x_fixed)]

            residual, _ = self.residual_fn(x_residual)
            residual_history.append(residual)
            logger.info(f"Residual Check — Previous: {past_residual:.6f}, Current: {residual:.6f}")

            if not residual < past_residual:
                logger.info(f"Residual did not decrease (delta = {residual - past_residual:.6f}). Stopping early.")
                break
            past_residual = residual

            # updating weights for regularization
            if regularizer:
                x_free_weights = np.delete(x_free, top_k_indices)
                regularizer.update_weights(x_free_weights)

        # fill in the remaining values (if any) with 0
        x_fixed[np.isnan(x_fixed)] = 0

        return A, x_fixed, residual_history

    def lsr(
        self,
        matrix_representation: MatrixRepresentation | None = "sparse",
        regularizer: RegularizationType | None = None,
        lambd: float | None = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, list[np.floating]]:
        """Solves a regularized least squares problem using quadratic programming.

        Parameters
        ----------
        matrix_representation : MatrixRepresentation or None, default="sparse"
            Format for matrix construction, e.g., "sparse" or "dense".
        regularizer : {"smooth", "abs"} or None, optional
            The type of regularization to apply. Supported values:
                - "smooth" : Applies smoothness regularization.
                - "abs" : Applies L1-norm (absolute value) regularization.
                - None or any other value : No regularization is applied.
        lambd : float, optional
            The regularization strength. Defaults to 0.1. Ignored if `regularizer` is None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[np.floating]]
            - The initial matrix of column vectors representing lines to be drawn.
            - The binary x solution of the system, where entries are either 1 or 0.
            - The residuals history.
        """
        logger.info(f"Regularized Least Squares: {regularizer}")

        regularization_map = {
            "smooth": SmoothRegularizer,
            "abs": AbsoluteValueRegularizer,
            # any other value -> NoRegularizer
        }

        regularizer_class = regularization_map.get(regularizer, NoRegularizer)
        regularizer_instance = regularizer_class()
        lambd = lambd if lambd else 0.1

        matrix_representation: MatrixRepresentation = matrix_representation if matrix_representation else "sparse"
        A = MatrixGenerator.compute_matrix(
            self.shape, self.number_of_pegs, self.crop_mode, matrix_representation, self.rasterization
        )

        x = self.solve_qp_cvxopt(A, self.b, regularizer_instance, lambd)

        residual = np.linalg.norm(self.b - A @ x)
        logger.info(f"Residual: {residual:.6f}")

        return A, x, [residual]

    def radon(
        self,
        patience: int | None = None,
        min_delta: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[np.floating]]:
        logger.info(f"Radon:")

        A_base = MatrixGenerator.compute_matrix(
            self.shape, self.number_of_pegs, self.crop_mode, "sparse", self.rasterization
        )
        A = A_base.copy()
        x = np.zeros(A_base.shape[1])
        k = A.shape[1]
        patience = patience if patience else 25
        min_delta = min_delta if min_delta else 1e-6

        radius, center_point = find_radius_and_center_point(self.shape, self.crop_mode)
        line_sizes = compute_line_lengths(self.number_of_pegs, radius, center_point)

        A_sum = A.sum(axis=1)
        A_sum_clipped = np.clip(np.array(A_sum).flatten(), 0.0, 1.0)
        sinogram = A_sum_clipped * self.b

        selected_lines = []
        past_residual = np.inf
        residual_history = []
        no_improvement_count = 0
        for step in tqdm(range(k), desc="Selecting Lines"):
            logger.info(f"Step {step + 1}/{k}")
            logger.info("-" * 30)

            A = A.tocsc()
            A_weighted = A.multiply(sinogram[:, None])
            col_sums = A_weighted.sum(axis=0).A1
            means = col_sums / line_sizes

            best_idx = np.argmax(means)
            best_line = A[:, best_idx].tocoo()

            selected_lines.append(best_idx)
            # divided by block_size from supersampling
            contribution = (A[:, best_idx].toarray().flatten() / self.block_size) * sinogram
            sinogram = np.clip(sinogram - contribution, 0, 1)

            A = A.tocsr()
            best_line_csr = best_line.tocsr()
            for i, val in zip(best_line_csr.indices, best_line_csr.data):
                row_start = A.indptr[i]
                row_end = A.indptr[i + 1]

                row_data = A.data[row_start:row_end]
                row_data -= val  # subtract val from all non-zero elements in row i

                # clip negatives to zero
                np.maximum(row_data, 0, out=row_data)

            A.eliminate_zeros()
            x[best_idx] = 1

            residual, _ = self.residual_fn(x)
            residual_history.append(residual)
            logger.info(f"Residual Check — Previous: {past_residual:.6f}, Current: {residual:.6f}")

            if past_residual - residual > min_delta:
                no_improvement_count = 0
                past_residual = residual
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    logger.info(f"Stopping early due to patience limit ({patience}).")
                    break

        return A_base, x, residual_history
