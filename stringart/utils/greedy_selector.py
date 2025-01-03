from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix

GreedySelector = Literal["random", "dot-product"]


class Selector(ABC):
    """Abstract base class for selecting top-k candidate lines based on a given matrix `A` and target vector `b`.

    Attributes
    ----------
    A : csr_matrix
        A sparse matrix representing candidate lines.
    b : np.ndarray
        A vector representing the target values.
    rows : int
        The number of rows in matrix `A`.
    cols : int
        The number of columns in matrix `A`.
    top_k : int
        The number of top candidates to select, based on `TOP_K_PERCENT`.

    Methods
    -------
    get_top_k_candidates()
        Abstract method for selecting the top-k candidate indices based on a certain heuristic.
    """

    TOP_K_PERCENT: float = 0.25

    def __init__(self, A: csr_matrix, b: np.ndarray):
        self.A = A
        self.b = b
        self.rows, self.cols = A.shape
        self.top_k = int(self.cols * Selector.TOP_K_PERCENT)

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


class RandomSelector(Selector):
    """Selects top-k candidates randomly from the matrix `A`."""

    def get_top_k_candidates(self):
        """Randomly selects the top-k candidate indices from the matrix `A`.

        Returns
        -------
        np.ndarray
           An array containing the indices of the randomly selected top-k candidates.
        """
        random_indices = np.random.choice(self.cols, self.top_k, replace=False)
        return random_indices


class DotProductSelector(Selector):
    """Selects top-k candidates based on the dot product between the columns of matrix `A` and the target vector `b`.

    Attributes
    ----------
    dot_products : np.ndarray
        An array containing the dot products between the columns of `A` and the vector `b`.
    """

    def __init__(self, A: csr_matrix, b: np.ndarray):
        super().__init__(A, b)
        self.dot_products = A.T @ b

    def get_top_k_candidates(self):
        """Selects the top-k candidate indices based on the highest absolute dot products between the columns of `A` and `b`.

        Returns
        -------
        np.ndarray
            An array containing the indices of the top-k candidates with the highest dot products.
        """

        candidate_indices = np.argsort(-np.abs(self.dot_products))
        top_candidates = candidate_indices[: self.top_k]

        return top_candidates
