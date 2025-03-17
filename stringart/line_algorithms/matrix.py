from typing import Callable, List

import numpy as np
from scipy.sparse import csr_matrix
from stringart.line_algorithms.bresenham import Bresenham
from stringart.utils.circle import compute_pegs
from stringart.utils.image import find_radius_and_center_point
from stringart.utils.types import Method, Mode, Point


class MatrixGenerator:
    """A utility class for generating a matrix representation of lines drawn between pegs placed within a 2D grid."""

    @staticmethod
    def compute_matrix(
        shape: tuple[int, ...], number_of_pegs: int, image_mode: Mode = "center", method: Method = "dense"
    ) -> tuple[np.ndarray | csr_matrix, List[Point]]:
        """Computes the matrix representation of lines drawn between pegs placed on a grid.

        Parameters
        ----------
        shape : tuple[int, ...]
            The dimensions of the grid (height, width) where the pegs will be placed.

        number_of_pegs : int
            The number of pegs to be placed on the grid.

        image_mode : Mode | None
            Specifies the location of the center point to start the peg arrangement. Can be one of:
            - "center" (default): Pegs are placed symmetrically around the center.
            - "first-half": Pegs are placed in the top-half/left-half portion of the rectangle.
            - "second-half": Pegs are placed in the bottom-half/right-half portion of the rectangle.

        method: Method
            The method used to generate the matrix. Can be "dense" or "sparse".

        Returns
        -------
        tuple[np.ndarray, List[Point], List[Point]]
            - A 2D numpy array (shape: number_of_lines x grid_size) where each row is a binary vector
              representing a line drawn between two pegs.
            - A list of Points representing the locations of the pegs.
        """
        radius, center_point = find_radius_and_center_point(shape, image_mode)
        pegs: List[Point] = compute_pegs(
            number_of_pegs=number_of_pegs,
            radius=radius,
            center_point=center_point,
        )

        if method not in MatrixGenerator.method_map:
            raise ValueError(f"Unknown method: {method}. Valid options are {list(MatrixGenerator.method_map.keys())}")

        A = MatrixGenerator.method_map[method](shape, pegs)

        return A, pegs

    @staticmethod
    def generate_dense_line(shape: tuple[int, ...], line: List[Point]) -> np.ndarray:
        """Generates a binary vector representing a line on the grid (the vector is in flattened matrix representation).

        The binary vector is of size `shape[0] * shape[1]` and each entry is set to `1` if the
        corresponding point is part of the line, and `0` otherwise.

        Parameters
        ----------
        shape : tuple[int, ...]
            The dimensions of the grid (height, width).

        line : List[Point]
            A list of Points representing the coordinates of the line's path between two pegs.

        Returns
        -------
        np.ndarray
            A 1D numpy array where each element represents a point in the grid. The value is `1` if the
            point is part of the line, otherwise `0`.
        """

        vector = np.zeros(shape[0] * shape[1], dtype=int)

        indices = [point.y * shape[1] + point.x for point in line]
        vector[indices] = 1

        return vector

    @staticmethod
    def generate_sparse_line(shape: tuple[int, ...], line: List[Point]) -> List[int]:
        """Generates a list of indices representing the positions of points in the line on a flattened grid.

        Parameters
        ----------
        shape : tuple[int, ...]
            The dimensions of the grid (height, width).

        line : List[Point]
            A list of Points representing the coordinates of the line's path between two pegs.

        Returns
        -------
        List[int]
            A list of indices representing the positions of points in the line on the flattened grid.
        """

        indices = [point.y * shape[1] + point.x for point in line]
        return indices

    @staticmethod
    def generate_dense_matrix(shape: tuple[int, ...], pegs: List[Point]) -> np.ndarray:
        """Generates a dense matrix representation of lines drawn between all pairs of pegs.

        Each column in the resulting matrix corresponds to a line between two pegs, represented as a binary vector
        in flattened grid form.

        Parameters
        ----------
        shape : tuple[int, ...]
            The dimensions of the grid (height, width).

        pegs : List[Point]
            A list of Points representing the coordinates of the pegs.

        Returns
        -------
        np.ndarray
            A 2D numpy array where each column is a binary vector representing a line between two pegs.
        """

        A = []

        for i in range(len(pegs)):
            for j in range(i + 1, len(pegs)):
                line = Bresenham.compute_line(pegs[i], pegs[j])

                vector = MatrixGenerator.generate_dense_line(shape, line)
                A.append(vector)

        return np.array(A).T

    @staticmethod
    def generate_sparse_matrix(shape: tuple[int, ...], pegs: List[Point]) -> csr_matrix:
        """Generates a sparse matrix representation of lines drawn between all pairs of pegs.

        The sparse matrix is in CSR format, where each non-zero entry corresponds to a point in the grid
        that is part of a line between two pegs.

        Parameters
        ----------
        shape : tuple[int, ...]
            The dimensions of the grid (height, width).

        pegs : List[Point]
            A list of Points representing the coordinates of the pegs.

        Returns
        -------
        scipy.sparse.csr_matrix
            A sparse matrix where each non-zero entry represents a point in the grid that is part of a line
            between two pegs.
        """

        row_indices: List[int] = []
        column_indices: List[int] = []
        column_index: int = 0

        for i in range(len(pegs)):
            for j in range(i + 1, len(pegs)):
                line = Bresenham.compute_line(pegs[i], pegs[j])

                sparse_line = MatrixGenerator.generate_sparse_line(shape, line)
                row_indices += sparse_line
                column_indices += [column_index] * len(sparse_line)

                column_index += 1

        A = csr_matrix(
            ([1.0] * len(row_indices), (row_indices, column_indices)), shape=(shape[0] * shape[1], column_index)
        )

        return A

    method_map: dict[str, Callable[[tuple[int, ...], list[Point]], np.ndarray | csr_matrix]] = {
        "dense": generate_dense_matrix,
        "sparse": generate_sparse_matrix,
    }
