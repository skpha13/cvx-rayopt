import numpy as np
from scipy.sparse import csr_matrix
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
    crop_mode : CropMode, optional
        Cropping mode for matrix generation. Default is "center".
    number_of_pegs : int, optional
        Number of pegs used in the matrix generation. Default is 100.
    rasterization : Rasterization, optional
        Line rasterization algorithm to use. Default is "bresenham".
    block_size : int
        Size of the square block used in downsampling via averaging.

    Attributes
    ----------
    b : np.ndarray
        Flattened and histogram-equalized version of the target image.
    """

    def __init__(
        self,
        image: np.ndarray,
        crop_mode: CropMode,
        number_of_pegs: int,
        rasterization: Rasterization,
        block_size: int = 8,
    ):
        self.shape = image.shape[:2]
        self.crop_mode: CropMode = crop_mode
        self.number_of_pegs = number_of_pegs
        self.rasterization: Rasterization = rasterization
        self.matrix_representation: MatrixRepresentation = "sparse"
        self.block_size = block_size if block_size else 8

        self.b: np.ndarray = ImageWrapper.histogram_equalization(image)
        self.b = ImageWrapper.flatten_image(self.b).astype(np.float64)

        self.upsample_shape = (self.shape[0] * block_size, self.shape[1] * block_size)
        self.A_upsampled = MatrixGenerator.compute_matrix(
            self.upsample_shape, self.number_of_pegs, self.crop_mode, self.matrix_representation, self.rasterization
        )

    def __call__(self, x: np.ndarray, *args, **kwargs) -> tuple[np.floating, np.ndarray]:
        """Apply the UDS transformation and compute residual error.

        Parameters
        ----------
        x : np.ndarray
            Solution vector representing the current estimate of line weights.

        Returns
        -------
        residual : float
            L2 norm between the downsampled result and the target image.
        solution : np.ndarray
            Downsampled grayscale image (after inversion and scaling to [0, 255]).
        """

        x_indices_set = kwargs.get("x_indices", None)

        if x_indices_set is not None:
            x_indices = np.array(list(x_indices_set))
            A = self.A_upsampled[:, x_indices]
        else:
            A = self.A_upsampled

        solution = A @ x
        solution = np.clip(np.reshape(solution, self.upsample_shape), a_min=0, a_max=1)

        blocks = view_as_blocks(solution, (self.block_size, self.block_size))
        DSA = blocks.mean(axis=(2, 3))

        residual = np.linalg.norm(self.b - ImageWrapper.flatten_image(DSA).astype(np.float64))

        solution = 1 - DSA
        solution = np.multiply(solution, 255).astype(np.uint8)

        return residual, solution
