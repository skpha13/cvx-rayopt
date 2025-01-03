import numpy as np
from numpy.linalg import lstsq
from scipy.sparse.linalg import lsqr

from stringart.line_algorithms.matrix import MatrixGenerator
from stringart.utils.image import ImageWrapper, crop_image
from stringart.utils.types import Method, Mode


class Solver:
    """A class to compute string art procedurally using multiple methods.

    Parameters
    ----------
    image : ImageWrapper
        An object representing the image to be processed.
    image_mode : Mode
        The mode in which the image is being processed. This determines cropping behaviour.
    """

    def __init__(self, image: ImageWrapper, image_mode: Mode):
        self.shape: tuple[int, ...] = image.get_shape()
        self.b: np.ndarray = image.flatten_image()
        self.image_mode: Mode = image_mode

    def least_squares(self, method: Method = "sparse") -> np.ndarray:
        """Solve the string art problem using the least squares method.

        Parameters
        ----------
        method : {'dense', 'sparse'}, optional
            The method to use for solving the least squares problem. Defaults to 'sparse'.
            - 'dense': Uses dense matrix operations via `numpy.linalg.lstsq`.
            - 'sparse': Uses sparse matrix operations via `scipy.sparse.linalg.lsqr`.

        Returns
        -------
        numpy.ndarray
            The reconstructed image as a 2D greyscale array.
            The pixel values are scaled between 0 and 255 and cropped according to the image mode.
        """

        A, pegs = MatrixGenerator.compute_matrix(self.shape, 100, self.image_mode, method)

        x = None
        if method == "dense":
            x, _, _, _ = lstsq(A, self.b)
        elif method == "sparse":
            x = lsqr(A, self.b)[0]

        solution = A @ x
        solution = np.clip(np.reshape(solution, shape=self.shape), a_min=0, a_max=1)
        solution = np.multiply(solution, 255).astype(np.uint8)
        solution = crop_image(solution, self.image_mode)

        return solution
