import pytest
from stringart.line_algorithms.xiaolin_wu import XiaolinWu
from stringart.utils.types import Point


class TestXiaolinWu:
    def test_ipart(self):
        assert XiaolinWu.ipart(3.7) == 3
        assert XiaolinWu.ipart(-3.7) == -4
        assert XiaolinWu.ipart(5.0) == 5

    def test_fpart(self):
        assert XiaolinWu.fpart(3.7) == pytest.approx(0.7, rel=1e-7)
        assert XiaolinWu.fpart(-3.7) == pytest.approx(0.3, rel=1e-7)
        assert XiaolinWu.fpart(5.0) == pytest.approx(0.0, rel=1e-7)

    def test_rfpart(self):
        assert XiaolinWu.rfpart(3.7) == pytest.approx(0.3, rel=1e-7)
        assert XiaolinWu.rfpart(-3.7) == pytest.approx(0.7, rel=1e-7)
        assert XiaolinWu.rfpart(5.0) == pytest.approx(1.0, rel=1e-7)

    def test_handle_endpoint(self):
        # Test the handle_endpoint method with both first and last endpoints
        point = Point(3.0, 4.0)
        gradient = 1.0  # simple diagonal line (45 degree)

        # Test the first endpoint handling
        indices, values = XiaolinWu.handle_endpoint(point, gradient, steep=False, is_first=True)
        assert len(indices) == 2
        assert len(values) == 2
        assert indices[0] == Point(3, 4)
        assert values[0] == pytest.approx(0.5)  # x_gap value should be around 0.5
        assert indices[1] == Point(3, 5)
        assert values[1] == pytest.approx(0.0)  # x_gap value should be around 0.0

        # Test the second endpoint handling
        indices, values = XiaolinWu.handle_endpoint(point, gradient, steep=False, is_first=False)
        assert len(indices) == 2
        assert len(values) == 2
        assert indices[0] == Point(3, 4)
        assert values[0] == pytest.approx(0.5)
        assert indices[1] == Point(3, 5)
        assert values[1] == pytest.approx(0.0)

    def test_compute_line(self):
        # Test the compute_line method with simple test cases
        point0 = Point(0, 0)
        point1 = Point(4, 4)
        indices, values = XiaolinWu.compute_line(point0, point1)

        # Check that the number of indices and values is correct
        assert len(indices) == 10  # The line should have 10 points
        assert len(values) == 10  # Corresponding 10 values

        # Check the values of some specific points
        assert indices[0] == Point(0, 0)
        assert values[0] == pytest.approx(0.5)  # The value at the first point is 0.5

        assert indices[4] == Point(1, 1)
        assert values[4] == pytest.approx(1.0)

        assert indices[8] == Point(3, 3)
        assert values[8] == pytest.approx(1.0)  # Value at the last point is 1.0

    def test_steep_line(self):
        # Test the line computation with a steep slope
        point0 = Point(0, 0)
        point1 = Point(2, 5)  # Steep line with a slope > 1
        indices, values = XiaolinWu.compute_line(point0, point1)

        # Check the result has expected number of points (line is steep)
        assert len(indices) == 12  # Expecting 12 points for this steep line
        assert len(values) == 12

        # Check a specific value for accuracy
        assert indices[0] == Point(0, 0)
        assert values[0] == pytest.approx(0.5)  # At the beginning, the value is 0.5
        assert indices[-1] == Point(2, 4)
        assert values[-1] == pytest.approx(0.6)  # End of the steep line, value should be 0.6

    def test_vertical_line(self):
        # Test the line computation with a vertical line
        point0 = Point(2, 0)  # x-coordinate is constant (2)
        point1 = Point(2, 5)  # y-coordinate varies (from 0 to 5)
        indices, values = XiaolinWu.compute_line(point0, point1)

        # Check that the line is vertical and includes all y-values from 0 to 5
        assert len(indices) == 6  # The line should have 6 points
        assert len(values) == 6  # Corresponding 6 values

        # Check that the x-coordinate is always 2, and y-values vary from 0 to 5
        for i, idx in enumerate(indices):
            assert idx.x == 2  # x should always be 2
            assert idx.y == point0.y + i  # y should increment by 1

        # Check the values for specific points
        assert values[0] == pytest.approx(1.0)  # First point at (2, 0)
        assert values[-1] == pytest.approx(1.0)  # Last point at (2, 5)

    def test_horizontal_line(self):
        # Test the line computation with a horizontal line
        point0 = Point(0, 3)  # y-coordinate is constant (3)
        point1 = Point(5, 3)  # x-coordinate varies (from 0 to 5)
        indices, values = XiaolinWu.compute_line(point0, point1)

        # Check that the line is horizontal and includes all x-values from 0 to 5
        assert len(indices) == 6  # The line should have 6 points
        assert len(values) == 6  # Corresponding 6 values

        # Check that the y-coordinate is always 3, and x-values vary from 0 to 5
        for i, idx in enumerate(indices):
            assert idx.y == 3  # y should always be 3
            assert idx.x == point0.x + i  # x should increment by 1

        # Check the values for specific points
        assert values[0] == pytest.approx(1.0)  # First point at (0, 3)
        assert values[-1] == pytest.approx(1.0)  # Last point at (5, 3)
