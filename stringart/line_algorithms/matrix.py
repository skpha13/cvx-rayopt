from typing import List

import numpy as np
from stringart.line_algorithms.bresenham import Bresenham
from stringart.utils.circle import compute_pegs
from stringart.utils.types import Point


class DenseMatrixGenerator:
    @staticmethod
    def compute_matrix(shape: tuple[int, int], number_of_pegs: int) -> tuple[np.ndarray, List[Point], List[Point]]:
        radius = min(shape[0], shape[1]) // 2
        pegs: List[Point] = compute_pegs(
            number_of_pegs=number_of_pegs,
            radius=radius,
            center_point=Point(radius, radius),
        )
        line_indices: List[Point] = []

        A = []
        for i in range(number_of_pegs):
            for j in range(i + 1, number_of_pegs):
                line = Bresenham.compute_line(pegs[i], pegs[j])
                line_indices.append(Point(i, j))

                vector = DenseMatrixGenerator.generate_line_matrix((shape[0] * shape[1], number_of_pegs), line)
                A.append(vector)

        return np.array(A), pegs, line_indices

    @staticmethod
    def generate_line_matrix(shape: tuple[int, int], line: List[Point]) -> np.ndarray:
        matrix = np.zeros(shape, dtype=int)

        indices = [point.y * shape[1] + point.x for point in line]
        matrix[indices] = 1

        return matrix.flatten()


# TODO: maybe have a DenseMatrixGenerator implementation and a SparseMatrixGenerator implementation
