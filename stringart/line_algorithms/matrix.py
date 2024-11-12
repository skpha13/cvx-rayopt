from typing import List

import numpy as np
from stringart.line_algorithms.bresenham import Bresenham
from stringart.utils.circle import compute_pegs
from stringart.utils.types import Point


class DenseMatrixGenerator:
    @staticmethod
    def compute_matrix(shape: tuple[int, int], number_of_pegs: int) -> tuple[np.ndarray, List[Point], List[Point]]:
        radius = min(shape[0], shape[1]) // 2 - 1
        pegs: List[Point] = compute_pegs(
            number_of_pegs=number_of_pegs,
            radius=radius,
            center_point=Point(radius + 1, radius + 1),
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
        vector = np.zeros(shape[0] * shape[1], dtype=int)

        indices = [point.y * shape[1] + point.x for point in line]
        vector[indices] = 1

        return vector


# TODO: maybe have a DenseMatrixGenerator implementation and a SparseMatrixGenerator implementation
