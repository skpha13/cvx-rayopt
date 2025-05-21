from abc import ABC, abstractmethod

import numpy as np


class Regularizer(ABC):
    """Abstract base class for regularization strategies."""

    @abstractmethod
    def prepare_matrices(self, P: np.ndarray, q: np.ndarray, n: int, lambd: float):
        """Prepare the P and q matrices along with the constraints G and h for the QP problem with regularization."""
        pass

    @abstractmethod
    def post_process(self, solution: dict, n: int) -> np.ndarray:
        """Post-process the solution if needed."""
        pass


class NoRegularizer(Regularizer):
    """No regularization - standard QP problem."""

    def prepare_matrices(
        self, P: np.ndarray, q: np.ndarray, n: int, lambd: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        G = np.vstack((-np.eye(n), np.eye(n)))
        h = np.hstack((np.zeros(n), np.ones(n)))

        return P, q, G, h

    def post_process(self, solution: dict, n: int) -> np.ndarray:
        return np.array(solution["x"]).flatten()


class SmoothRegularizer(Regularizer):
    """Smooth regularization that encourages values between 0 and 1. 4x(1-x)"""

    def prepare_matrices(
        self, P: np.ndarray, q: np.ndarray, n: int, lambd: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Add -8*lambda to diagonal of P because P in qp is 2x of Hessian
        min_eig = np.min(np.linalg.eigvalsh(P))
        max_lambda = min_eig / 8
        if lambd > max_lambda:
            # TODO: use logger instead
            print(f"Warning: lambd too large for PSD: reducing lambd from {lambd} to {max_lambda * 0.9}")
            lambd = max_lambda * 0.9

        P_reg = P + np.eye(n) - 8 * lambd * np.eye(n)
        q_reg = q + 4 * lambd * np.ones(n)

        G = np.vstack((-np.eye(n), np.eye(n)))
        h = np.hstack((np.zeros(n), np.ones(n)))

        return P_reg, q_reg, G, h

    def post_process(self, solution: dict, n: int) -> np.ndarray:
        return np.array(solution["x"]).flatten()


class AbsoluteValueRegularizer(Regularizer):
    """Absolute value regularization that encourages values to be close to 0.5. -2|x-0.5|"""

    def prepare_matrices(
        self, P: np.ndarray, q: np.ndarray, n: int, lambd: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        P_block = np.block([[P, np.zeros((n, n))], [np.zeros((n, n)), np.zeros((n, n))]])
        q_block = np.hstack([q, -2 * lambd * np.ones(n)])

        G_upper = np.vstack(
            [
                np.hstack([-np.eye(n), np.zeros((n, n))]),  # -x <= 0
                np.hstack([np.eye(n), np.zeros((n, n))]),  # x <= 1
                np.hstack([np.eye(n), -np.eye(n)]),  # x - t <= 0.5
                np.hstack([-np.eye(n), -np.eye(n)]),  # -x - t <= -0.5
            ]
        )
        h_upper = np.hstack([np.zeros(n), np.ones(n), 0.5 * np.ones(n), -0.5 * np.ones(n)])

        return P_block, q_block, G_upper, h_upper

    def post_process(self, solution: dict, n: int) -> np.ndarray:
        x = np.array(solution["x"][:n]).flatten()
        return x


class BinaryValueRegularizer(Regularizer):
    """Regularization that encourages values to be exactly 0 or 1."""

    def prepare_matrices(
        self, P: np.ndarray, q: np.ndarray, n: int, lambd: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        P_block = np.block([[P, np.zeros((n, n))], [np.zeros((n, n)), np.zeros((n, n))]])
        q_block = np.hstack([q, lambd * np.ones(n)])

        G_upper = np.vstack(
            [
                np.hstack([-np.eye(n), np.zeros((n, n))]),  # -x <= 0
                np.hstack([np.eye(n), np.zeros((n, n))]),  # x <= 1
                np.hstack([np.eye(n), -np.eye(n)]),  # x - t <= 0
                np.hstack([-np.eye(n), -np.eye(n)]),  # -x - t <= -1
            ]
        )
        h_upper = np.hstack([np.zeros(n), np.ones(n), np.zeros(n), -np.ones(n)])

        return P_block, q_block, G_upper, h_upper

    def post_process(self, solution: dict, n: int) -> np.ndarray:
        x = np.array(solution["x"][:n]).flatten()
        return x
