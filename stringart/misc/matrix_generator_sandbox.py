import numpy as np
import scipy
from stringart.line_algorithms.matrix import DenseMatrixGenerator
from stringart.utils.image import ImageWrapper

image = ImageWrapper()
image.read_bw("../../imgs/lenna.png")
b = image.flatten_image()

shape = image.get_shape()
A, pegs, lines = DenseMatrixGenerator.compute_matrix(shape, 10)

first_line = A[:, 2]
matrix = np.reshape(first_line, shape=shape)
pegs_arr = [[point.y, point.x] for point in pegs]

for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if [i, j] in pegs_arr:
            print("*", end=" ")
            continue
        print("." if matrix[i][j] == 0 else "#", end=" ")
    print()

x, _, _, _ = scipy.linalg.lstsq(A, b)

print(x)
