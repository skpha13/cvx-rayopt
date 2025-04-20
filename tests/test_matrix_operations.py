import numpy as np
import pytest
from stringart.line_algorithms.matrix import MatrixGenerator
from stringart.utils.circle import compute_pegs
from stringart.utils.image import crop_image, find_radius_and_center_point
from stringart.utils.types import Point


class TestMatrixOperations:
    @pytest.mark.parametrize(
        "shape",
        [
            (199, 199),
            (100, 100),
            (101, 101),
            (120, 100),
            (100, 120),
            (121, 100),
            (100, 121),
        ],
    )
    @pytest.mark.parametrize("crop_mode", ["center", "first-half", "second-half"])
    def test_cropped_image_shape_and_matrix_alignment(self, shape, crop_mode):
        image = np.zeros(shape, dtype=np.uint8)

        cropped = crop_image(image, crop_mode)
        A, pegs = MatrixGenerator.compute_matrix(
            shape=np.shape(cropped),
            number_of_pegs=20,
            crop_mode=crop_mode,
            matrix_representation="dense",
            rasterization="bresenham",
        )

        assert (
            A.shape[0] == cropped.size
        ), f"Matrix row size {A.shape[0]} does not match cropped image size {cropped.size}"
        assert A.shape[1] == (20 * 19) // 2, "Matrix column count should match number of peg-pairs"
        assert min(cropped.shape) == min(image.shape)

    def test_find_radius_and_center_point_values(self):
        shape = (100, 80)

        for mode in ["center", "first-half", "second-half"]:
            radius, center = find_radius_and_center_point(shape, mode)

            assert isinstance(radius, int) and radius > 0

            if mode is not None:
                assert isinstance(center, Point)
                assert 0 <= center.x < shape[1]
                assert 0 <= center.y < shape[0]

    def test_generate_dense_matrix_output_dimensions(self):
        shape = (100, 100)
        pegs = compute_pegs(10, 40, Point(50, 50))

        matrix = MatrixGenerator.generate_dense_matrix(shape, pegs, rasterization="bresenham")
        assert matrix.shape == (shape[0] * shape[1], 45)  # 10 pegs → 45 pairs

    def test_generate_dense_line_correct_vector_shape(self):
        shape = (20, 30)
        line = [Point(x, x) for x in range(10)]
        vector = MatrixGenerator.generate_dense_line(shape, line)
        assert vector.shape == (20 * 30,)
        assert np.sum(vector) == len(line)
