from typing import Tuple

import numpy as np


def D(N: int, w: int) -> float:
    """Compute the approximate Manhattan distance between two adjacent pegs
    on a circular arrangement.

    Parameters
    ----------
    N : int
        Number of pegs arranged around the circle.
    w : int
        Width of the square image (used to compute the radius).

    Returns
    -------
    float
        Manhattan distance between two neighboring pegs on the circle.
    """
    return w // 2 * (np.abs(np.cos(2 * np.pi / N) - 1) + np.abs(np.sin(2 * np.pi / N)))


def find_optimal_pegs(shape: Tuple[int, ...]) -> int:
    """Find the optimal number of pegs based on image shape and a target Manhattan distance.

    Parameters
    ----------
    shape : tuple of int
        Shape of the image, typically (height, width). Only the width (shape[1]) is used.

    Returns
    -------
    int
        Optimal number of pegs that minimizes the difference between the computed
        Manhattan distance and a predefined threshold.
    """

    # obtained from string.experiments.peg_optimization_analysis.py
    THRESHOLD: float = 10.0
    pegs = [32, 64, 128, 256, 512]

    d = np.inf
    N = 0
    w = shape[1]
    for n in pegs:
        if np.abs(D(n, w) - THRESHOLD) < np.abs(d - THRESHOLD):
            d = D(n, w)
            N = n

    return N


if __name__ == "__main__":
    shape = (330, 330)
    number_of_pegs = find_optimal_pegs(shape)
    print(f"Shape: {shape}\nOptimal Number of Pegs: {number_of_pegs}")
