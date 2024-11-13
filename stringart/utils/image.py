import numpy as np
from imageio.v3 import imread
from skimage.color import rgb2gray


class ImageWrapper:
    def __init__(self) -> None:
        self.image: np.ndarray | None = None

    def check_image_loaded(self) -> None:
        if self.image is None:
            raise ValueError("Image cannot be None. Most probably it was not loaded.")

    def read_bw(self, file_path: str) -> None:
        self.image = rgb2gray(imread(file_path))

    def flatten_image(self) -> np.ndarray:
        self.check_image_loaded()

        return self.image.flatten()

    def get_shape(self) -> tuple[int, int]:
        self.check_image_loaded()

        return self.image.shape
