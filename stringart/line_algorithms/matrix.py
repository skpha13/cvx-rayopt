from typing import List, Literal

import numpy as np
from stringart.line_algorithms.bresenham import Bresenham
from stringart.utils.circle import compute_pegs
from stringart.utils.types import Point


class DenseMatrixGenerator:
    """A utility class for generating a dense matrix representation of lines drawn between pegs placed within a 2D grid."""

    @staticmethod
    def compute_matrix(
        shape: tuple[int, int],
        number_of_pegs: int,
        mode: Literal["first-half", "center", "second-half"] = "center",
    ) -> tuple[np.ndarray, List[Point], List[Point]]:
        """Computes the dense matrix representation of lines drawn between pegs placed on a grid.

        Parameters:
        -----------
        shape : tuple[int, int]
            The dimensions of the grid (height, width) where the pegs will be placed.

        number_of_pegs : int
            The number of pegs to be placed on the grid.

        mode : Literal["first-half", "center", "second-half"], optional
            Specifies the location of the center point to start the peg arrangement. Can be one of:
            - "center" (default): Pegs are placed symmetrically around the center.
            - "first-half": Pegs are placed in the top-half/left-half portion of the rectangle.
            - "second-half": Pegs are placed in the bottom-half/right-half portion of the rectangle.

        Returns:
        --------
        tuple[np.ndarray, List[Point], List[Point]]:
            - A 2D numpy array (shape: number_of_lines x grid_size) where each row is a binary vector
              representing a line drawn between two pegs.
            - A list of Points representing the locations of the pegs.
            - A list of Point pairs representing the indices of pegs that are connected by a line.
        """

        radius = min(shape[0], shape[1]) // 2 - 1

        center_point = None
        if mode == "center":
            center_point = Point(radius + 1, shape[0] // 2) if shape[0] > shape[1] else Point(shape[1] // 2, radius + 1)
        elif mode == "first-half":
            center_point = Point(radius + 1, radius + 1)
        elif mode == "second-half":
            center_point = (
                Point(radius + 1, shape[0] - radius - 1)
                if shape[0] > shape[1]
                else Point(shape[1] - radius - 1, radius + 1)
            )

        pegs: List[Point] = compute_pegs(
            number_of_pegs=number_of_pegs,
            radius=radius,
            center_point=center_point,
        )
        line_indices: List[Point] = []

        A = []
        for i in range(number_of_pegs):
            for j in range(i + 1, number_of_pegs):
                line = Bresenham.compute_line(pegs[i], pegs[j])
                line_indices.append(Point(i, j))

                vector = DenseMatrixGenerator.generate_line_matrix(shape, line)
                A.append(vector)

        return np.array(A).T, pegs, line_indices

    @staticmethod
    def generate_line_matrix(shape: tuple[int, int], line: List[Point]) -> np.ndarray:
        """Generates a binary vector representing a line on the grid (the vector is in flattened matrix representation).

        The binary vector is of size `shape[0] * shape[1]` and each entry is set to `1` if the
        corresponding point is part of the line, and `0` otherwise.

        Parameters:
        -----------
        shape : tuple[int, int]
            The dimensions of the grid (height, width).

        line : List[Point]
            A list of Points representing the coordinates of the line's path between two pegs.

        Returns:
        --------
        np.ndarray:
            A 1D numpy array where each element represents a point in the grid. The value is `1` if the
            point is part of the line, otherwise `0`.
        """

        vector = np.zeros(shape[0] * shape[1], dtype=int)

        indices = [point.y * shape[1] + point.x for point in line]
        vector[indices] = 1

        return vector
