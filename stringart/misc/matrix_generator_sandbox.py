from stringart.line_algorithms.matrix import DenseMatrixGenerator

A, pegs, lines = DenseMatrixGenerator.compute_matrix((100, 100), 10)

print(A)
