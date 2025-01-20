from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr
from sklearn.preprocessing import normalize
from stringart.utils.greedy_selector import DotProductSelector, GreedySelector, RandomSelector, Selector


class MatchingPursuit(ABC):
    """Abstract base class for matching pursuit algorithms.

    Parameters
    ----------
    A : csr_matrix
        The matrix representing the system of equations.
    b : np.ndarray
        The target vector in the system of equations.
    """

    def __init__(self, A: csr_matrix, b: np.ndarray):
        self.A = A
        self.b = b.copy()

    @abstractmethod
    def compute_best_column(self, remaining_candidate_lines: csr_matrix) -> int:
        """Abstract method for computing the best column index from the remaining candidate lines.

        Parameters
        ----------
        remaining_candidate_lines : csr_matrix
            The matrix containing the candidate columns to consider.

        Returns
        -------
        int
            The index of the best column.
        """
        pass


class Greedy(MatchingPursuit):
    """Greedy matching pursuit algorithm using a selector.

    Parameters
    ----------
    A : csr_matrix
        The matrix representing the system of equations.
    b : np.ndarray
        The target vector in the system of equations.
    selector_type : GreedySelector, optional
        The type of selector to use. It should be one of 'random' or 'dot-product'.
        Defaults to 'random'.

    Attributes
    ----------
    selector_type : GreedySelector
        The type of selector used to select top k candidates.
    """

    __greedy_map_selector: dict[str, type[Selector]] = {"random": RandomSelector, "dot-product": DotProductSelector}

    def __init__(self, A: csr_matrix, b: np.ndarray, selector_type: GreedySelector = "random"):
        super().__init__(A, b)
        self.selector_type = selector_type

    def compute_best_column(self, remaining_candidate_lines: csr_matrix) -> int:
        """Computes the best column to add to the solution using a greedy approach.

        This method selects the best column from the remaining candidates based on a
        specified selection strategy.

        Parameters
        ----------
        remaining_candidate_lines : csr_matrix
           The matrix containing the remaining candidate columns.

        Returns
        -------
        int
           The index of the best column to add to the solution.

        Raises
        ------
        ValueError
           If no valid column is found.
        """

        # initialize selector
        selector: Selector = Greedy.__greedy_map_selector[self.selector_type](
            remaining_candidate_lines,
            self.b,
        )

        best_index = -1
        best_residual = np.inf
        top_candidates = selector.get_top_k_candidates()

        for column_index in top_candidates:
            trial_column = remaining_candidate_lines[:, column_index]
            A_trial = hstack([self.A, trial_column])
            x_trial = lsqr(A_trial, self.b)[0]

            residual = np.linalg.norm(self.b - A_trial @ x_trial)

            if residual < best_residual:
                best_index = column_index
                best_residual = residual

        # if no line was found to be drawn
        if best_index == -1:
            raise ValueError("No line found to be drawn")

        # update A matrix
        self.A = hstack([self.A, remaining_candidate_lines[:, best_index]])

        return best_index


class Orthogonal(MatchingPursuit):
    """Orthogonal matching pursuit algorithm.

    Attributes
    ----------
    A : csr_matrix
        The matrix representing the system of equations.
    b : np.ndarray
        The target vector.
    """

    def compute_best_column(self, remaining_candidate_lines: csr_matrix) -> int:
        """Computes the best column to add to the solution using an orthogonal matching pursuit strategy.

        This method finds the column that has the highest correlation with the residual
        vector and then updates the residual.

        Parameters
        ----------
        remaining_candidate_lines : csr_matrix
            The matrix containing the remaining candidate columns.

        Returns
        -------
        int
            The index of the best column to add to the solution.
        """

        # normalize column vectors
        A_norm = normalize(remaining_candidate_lines, norm="l2", axis=0, copy=True)

        dot_products = A_norm.T @ self.b
        best_index = np.argmax(dot_products)
        self.b = self.b - A_norm[:, best_index].toarray().ravel()  # to flatten the array as it comes in matrix form

        return best_index
