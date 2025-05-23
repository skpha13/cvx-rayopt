import numpy as np
from skimage.util import view_as_blocks
from stringart.line.matrix import MatrixGenerator
from stringart.utils.image import ImageWrapper
from stringart.utils.types import CropMode, MatrixRepresentation, Rasterization


class UDSLoss:
    """Upsample-Downsample (UDS) Loss Operator.

    This class performs a transformation that upsamples a linear solution vector,
    then downsamples it via block-wise averaging. It compares the downsampled
    result to a target image and returns both the processed image and a residual
    (L2 norm) error.

    Parameters
    ----------
    image : np.ndarray
        Target grayscale image (2D array) to compare against after downsampling.
    matrix_representation : MatrixRepresentation, optional
        Type of matrix representation to use for the upsampling operation.
        Default is "sparse".
    crop_mode : CropMode, optional
        Cropping mode for matrix generation. Default is "center".
    number_of_pegs : int, optional
        Number of pegs used in the matrix generation. Default is 100.
    rasterization : Rasterization, optional
        Line rasterization algorithm to use. Default is "bresenham".

    Attributes
    ----------
    b : np.ndarray
        Flattened and histogram-equalized version of the target image.
    """

    def __init__(
        self,
        image: np.ndarray,
        matrix_representation: MatrixRepresentation | None = "sparse",
        crop_mode: CropMode | None = "center",
        number_of_pegs: int | None = 100,
        rasterization: Rasterization | None = "bresenham",
    ):
        self.crop_mode: CropMode = crop_mode if crop_mode else "center"
        self.number_of_pegs = number_of_pegs if number_of_pegs else 100
        self.rasterization: Rasterization = rasterization if rasterization else "bresenham"
        self.matrix_representation: MatrixRepresentation = matrix_representation if matrix_representation else "sparse"

        self.b: np.ndarray = ImageWrapper.histogram_equalization(image)
        self.b = ImageWrapper.flatten_image(self.b).astype(np.float64)

    def __call__(self, x: np.ndarray, shape: tuple[int, int], block_size: int, *args, **kwargs):
        """Apply the UDS transformation and compute residual error.

        Parameters
        ----------
        x : np.ndarray
            Solution vector representing the current estimate of line weights.
        shape : tuple of int
            Target image shape before upsampling (height, width).
        block_size : int
            Size of the square block used in downsampling via averaging.

        Returns
        -------
        solution : np.ndarray
            Downsampled grayscale image (after inversion and scaling to [0, 255]).
        residual : float
            L2 norm between the downsampled result and the target image.
        """

        upsample_shape = (shape[0] * block_size, shape[1] * block_size)
        A_upsampled, _ = MatrixGenerator.compute_matrix(
            upsample_shape, self.number_of_pegs, self.crop_mode, self.matrix_representation, self.rasterization
        )

        solution = A_upsampled @ x
        solution = np.clip(np.reshape(solution, upsample_shape), a_min=0, a_max=1)

        blocks = view_as_blocks(solution, (block_size, block_size))
        downsampled = blocks.mean(axis=(2, 3))

        residual = np.linalg.norm(self.b - ImageWrapper.flatten_image(downsampled).astype(np.float64))

        solution = 1 - downsampled
        solution = np.multiply(solution, 255).astype(np.uint8)

        return solution, residual
