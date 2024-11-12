import numpy as np
from stringart.line_algorithms.matrix import DenseMatrixGenerator

A, pegs, lines = DenseMatrixGenerator.compute_matrix((20, 20), 10)

first_line = A[:, 2]
matrix = np.reshape(first_line, shape=(20, 20))
pegs_arr = [[point.y, point.x] for point in pegs]

for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if [i, j] in pegs_arr:
            print("*", end=" ")
            continue
        print("." if matrix[i][j] == 0 else "#", end=" ")
    print()
