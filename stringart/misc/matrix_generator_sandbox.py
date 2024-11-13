import numpy as np
import scipy
from matplotlib import pyplot as plt
from skimage import io
from stringart.line_algorithms.matrix import DenseMatrixGenerator
from stringart.utils.image import ImageWrapper

image = ImageWrapper()
image.read_bw("../../imgs/lena.png")
b = image.flatten_image()

shape = image.get_shape()
A, pegs, lines = DenseMatrixGenerator.compute_matrix(shape, 100)

first_line = A[:, 2]
matrix = np.reshape(first_line, shape=shape)
pegs_arr = [[point.y, point.x] for point in pegs]

# for i in range(len(matrix)):
#     for j in range(len(matrix[i])):
#         if [i, j] in pegs_arr:
#             print("*", end=" ")
#             continue
#         print("." if matrix[i][j] == 0 else "#", end=" ")
#     print()

x, _, _, _ = scipy.linalg.lstsq(A, b)

solution = np.dot(A, x)
solution = np.clip(np.reshape(solution, shape=shape), a_min=0, a_max=1)
solution = np.multiply(solution, 255).astype(np.uint8)

io.imsave("../../outputs/lena_stringart.png", solution)
io.imshow(solution)
plt.show()
