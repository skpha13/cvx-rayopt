from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from scipy.sparse import csc_matrix
from sklearn.preprocessing import normalize

GreedySelector = Literal["random", "dot-product", "all"]


class Selector(ABC):
    """Abstract base class for selecting top-k candidate lines based on a given matrix `A` and target vector `b`.

    Attributes
    ----------
    A : csc_matrix
        A sparse matrix representing candidate lines.
    b : np.ndarray
        A vector representing the target values.
    rows : int
        The number of rows in matrix `A`.
    cols : int
        The number of columns in matrix `A`.
    TOP_K : int
        The number of top candidates to select.

    Methods
    -------
    get_top_k_candidates()
        Abstract method for selecting the top-k candidate indices based on a certain heuristic.
    """

    TOP_K: int = 100

    def __init__(self, A: csc_matrix, b: np.ndarray):
        self.A = A
        self.b = b
        self.rows, self.cols = A.shape
        Selector.TOP_K = min(Selector.TOP_K, self.cols)

    @abstractmethod
    def get_top_k_candidates(self):
        """Selects the top-k candidate indices based on a certain heuristic.

        This method should be implemented in subclasses to define how the candidates are selected.

        Returns
        -------
        np.ndarray
            An array containing the indices of the top-k selected candidates.
        """
        pass


class AllSelector(Selector):
    """Selects all candidates from the matrix `A`."""

    def get_top_k_candidates(self):
        return np.arange(self.cols)


class RandomSelector(Selector):
    """Selects top-k candidates randomly from the matrix `A`."""

    def get_top_k_candidates(self):
        """Randomly selects the top-k candidate indices from the matrix `A`.

        Returns
        -------
        np.ndarray
           An array containing the indices of the randomly selected top-k candidates.
        """
        random_indices = np.random.choice(range(self.cols), Selector.TOP_K, replace=False)
        return random_indices


class DotProductSelector(Selector):
    """Selects top-k candidates based on the dot product between the columns of matrix `A` and the target vector `b`.

    Attributes
    ----------
    dot_products : np.ndarray
        An array containing the dot products between the columns of `A` and the vector `b`.
    """

    def __init__(self, A: csc_matrix, b: np.ndarray):
        super().__init__(A, b)

        A_norm = normalize(A, norm="l2", axis=0, copy=True)
        self.dot_products = A_norm.T @ b

    def get_top_k_candidates(self):
        """Selects the top-k candidate indices based on the highest absolute dot products between the columns of `A` and `b`.

        Returns
        -------
        np.ndarray
            An array containing the indices of the top-k candidates with the highest dot products.
        """

        candidate_indices = np.argsort(-np.abs(self.dot_products))
        top_candidates = candidate_indices[: Selector.TOP_K]

        return top_candidates
