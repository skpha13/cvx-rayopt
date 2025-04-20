from pathlib import Path

import numpy as np
from matplotlib.pyplot import imread
from skimage.color import rgb2gray
from stringart.utils.types import CropMode, Point


class ImageWrapper:
    @staticmethod
    def read_bw(file_path: str | Path, inverted: bool = True) -> np.ndarray:
        image = imread(file_path)

        # remove alpha channel
        if image.shape[-1] == 4:
            image = image[..., :3]

        if len(image.shape) >= 3:
            image = rgb2gray(image)

        if inverted:
            return 1 - image  # inverting black with white

        return image

    @staticmethod
    def flatten_image(image: np.ndarray) -> np.ndarray:
        return image.flatten()

    @staticmethod
    def scale_image(image: np.ndarray) -> np.ndarray:
        """Scale an image to the range [0, 1].

        This function scales the pixel values of an input image to lie between 0 and 1.
        The scaling is performed using the formula:
            scaled_image = (image - min_value) / (max_value - min_value)
        where `min_value` and `max_value` are the minimum and maximum values of the image, respectively.

        Parameters
        ----------
        image : np.ndarray
            The input image represented as a NumPy array. It can have any shape,
            such as (H, W) for grayscale images or (H, W, C) for color images.

        Returns
        -------
        np.ndarray
            A NumPy array of the same shape as the input, with values normalized to the range [0, 1].
        """

        min_value = np.min(image)
        max_value = np.max(image)
        return (image - min_value) / (max_value - min_value)

    @staticmethod
    def alpha_map(image: np.ndarray, crop_mode: CropMode) -> np.ndarray:
        """Generates an alpha map based on the center and radius derived from the image dimensions and mode.

        Parameters
        ----------
        image : np.ndarray
            A NumPy array representing the image. Its shape should be in the format (height, width, channels) or (height, width).
        crop_mode : CropMode
            Image cropping mode that will be used on the image.

        Returns
        -------
        np.ndarray
            A 2D NumPy array (height, width) representing the alpha map, where each pixel value is either 0.0 or 1.0,
            indicating transparency (0) or full opacity (1) based on the distance from the center point.

        Raises
        ------
        ValueError
            If the image shape is not compatible with the Mode.
        """

        radius, center_point = find_radius_and_center_point(image.shape, crop_mode)
        alpha_map = np.zeros(image.shape[:2])

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                distance = np.sqrt((x - center_point.x) ** 2 + (y - center_point.y) ** 2)

                if distance <= radius:
                    alpha_map[y, x] = 1.0

        return alpha_map

    @staticmethod
    def apply_alpha_map_bw_to_rgba(bw_image: np.ndarray, alpha_map: np.ndarray) -> np.ndarray:
        """Applies an alpha map to a black-and-white image, converting it into an RGBA image.

        Parameters
        ----------
        bw_image : np.ndarray
            A NumPy array representing a black-and-white image with shape (height, width).
        alpha_map : np.ndarray
            A 2D NumPy array representing the alpha map, with the same height and width as the black-and-white image.

        Returns
        -------
        np.ndarray
            A 3D NumPy array of shape (height, width, 4) representing the RGBA image, where the first three channels
            (red, green, and blue) are set to the grayscale values of the black-and-white image, and the fourth channel
            (alpha) is set based on the alpha map.

        Raises
        ------
        ValueError
            If the black-and-white image and the alpha map do not have the same dimensions.
        """

        if bw_image.shape != alpha_map.shape:
            raise ValueError("The alpha map and the black-and-white image must have the same dimensions.")

        rgba_image = np.zeros((bw_image.shape[0], bw_image.shape[1], 4))

        rgba_image[..., 0] = bw_image  # red channel
        rgba_image[..., 1] = bw_image  # green channel
        rgba_image[..., 2] = bw_image  # blue channel
        rgba_image[..., 3] = alpha_map  # alpha channel

        return rgba_image

    @staticmethod
    def histogram_equalization(src: np.ndarray) -> np.ndarray:
        """Perform histogram equalization on a grayscale image.

        Histogram equalization improves the contrast of the image by
        spreading out the most frequent intensity values.

        Parameters
        ----------
        src : np.ndarray
            A grayscale image represented as a 2D NumPy array with
            pixel intensity values ranging from 0 to 255.

        Returns
        -------
        np.ndarray
            The equalized grayscale image with enhanced contrast.

        Notes
        -------

        See link for more details: https://docs.opencv.org/4.x/d4/d1b/tutorial_histogram_equalization.html
        """
        src = (src * 255).astype(np.uint8)
        hist, bins = np.histogram(src.flatten(), 256, (0, 256))

        cdf = hist.cumsum()

        # normalize the cdf while avoiding zeros
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype("uint8")

        dst = cdf[src]
        dst = dst.astype(np.float32) / 255.0

        return dst

    @staticmethod
    def grayscale_quantization(src: np.ndarray, n_levels: int = 2) -> np.ndarray:
        """Perform grayscale quantization on an image with a specified number of quantization levels.

        Parameters
        ----------
        src : np.ndarray
            A grayscale image represented as a 2D NumPy array with pixel values in the range [0, 1].

        n_levels : int, optional
            The number of quantization levels. The default is 2.

        Returns
        -------
        np.ndarray
            A grayscale image with quantized pixel values in the range [0, 1].
        """

        src = (src * 255).astype(np.uint8)

        delta = 255 / n_levels
        dst = np.floor(src / delta) * delta + delta / 2
        dst = dst.astype(np.float32) / 255.0

        return dst


def find_radius_and_center_point(shape: tuple[int, ...], crop_mode: CropMode | None = None) -> tuple[int, Point | None]:
    """Calculate the radius and center point of a region within an image shape.

    Parameters
    ----------
        shape : (tuple[int, ...])
            The dimensions of the image (height, width).

        crop_mode : CropMode | None
            Specifies the location of the center point to start the peg arrangement. Can be one of:
            - "center" (default): Pegs are placed symmetrically around the center.
            - "first-half": Pegs are placed in the top-half/left-half portion of the rectangle.
            - "second-half": Pegs are placed in the bottom-half/right-half portion of the rectangle.

    Returns
    -------
        radius, center_point : tuple[int, Point | None]
            - radius (int): Half the smaller dimension of the image (height or width).
            - center_point (Point | None): A Point object representing the center, or None if mode is None.
    """
    radius = min(shape[0], shape[1]) // 2 - 1

    if crop_mode is None:
        return radius, None

    center_point = None
    if crop_mode == "center":
        center_point = Point(radius, shape[0] // 2) if shape[0] > shape[1] else Point(shape[1] // 2, radius)
    elif crop_mode == "first-half":
        center_point = Point(radius, radius)
    elif crop_mode == "second-half":
        center_point = Point(radius, shape[0] - radius) if shape[0] > shape[1] else Point(shape[1] - radius, radius)

    return radius, center_point


def crop_image(image: np.ndarray, crop_mode: CropMode) -> np.ndarray:
    """Crop an image to a square by its minimum length using the specified cropping mode.

    Parameters
    ----------
    image : np.ndarray
        The input image as a NumPy array of shape (H, W, C) or (H, W).
    crop_mode : CropMode
        The cropping mode to use. Must be one of:
        - "center": crop the square from the center of the image.
        - "first-half": crop the square from the top-left corner.
        - "second-half": crop the square from the bottom-right corner.

    Returns
    -------
    np.ndarray
        The cropped square image as a NumPy array of shape (L, L) or (L, L, C),
        where L is the minimum of the original height and width.
    """

    height, width = image.shape[:2]
    min_length = min(height, width)

    if crop_mode == "center":
        top = (height - min_length) // 2
        left = (width - min_length) // 2
    elif crop_mode == "first-half":
        top = 0
        left = 0
    elif crop_mode == "second-half":
        top = height - min_length
        left = width - min_length

    return image[top : top + min_length, left : left + min_length]
