from pathlib import Path

import numpy as np
from imageio.v3 import imread
from skimage.color import rgb2gray
from stringart.utils.types import Mode, Point


class ImageWrapper:
    @staticmethod
    def read_bw(file_path: str | Path) -> np.ndarray:
        grayscale_image = rgb2gray(imread(file_path))
        return 1 - grayscale_image  # inverting black with white

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


def find_radius_and_center_point(shape: tuple[int, ...], mode: Mode | None = None) -> tuple[int, Point | None]:
    """Calculate the radius and center point of a region within an image shape.

    Parameters
    ----------
        shape : (tuple[int, ...])
            The dimensions of the image (height, width).

        mode : Mode | None
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

    if mode is None:
        return radius, None

    center_point = None
    if mode == "center":
        center_point = Point(radius, shape[0] // 2) if shape[0] > shape[1] else Point(shape[1] // 2, radius)
    elif mode == "first-half":
        center_point = Point(radius, radius)
    elif mode == "second-half":
        center_point = Point(radius, shape[0] - radius) if shape[0] > shape[1] else Point(shape[1] - radius, radius)

    return radius, center_point


def crop_image(image: np.ndarray, mode: Mode) -> np.ndarray:
    """Crops an input image according to a specified mode.

    Parameters
    ----------
        image : np.ndarray
            The input image array (height, width, channels).
        mode : Mode
            The cropping mode.
            Options are:
                - "center": Crop around the geometric center of the image.
                - "first-half": Crop around the top-left region.
                - "second-half": Crop around the bottom-right region.

    Returns
    -------
        cropped_image : np.ndarray
            The cropped square region of the image.
    """
    shape = np.shape(image)
    radius, center_point = find_radius_and_center_point(shape, mode)

    top = center_point.y - radius
    left = center_point.x - radius
    bottom = center_point.y + radius
    right = center_point.x + radius

    cropped_image = image[top:bottom, left:right]
    return cropped_image
