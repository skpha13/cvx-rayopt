from stringart.line_algorithms.bresenham import Bresenham, Point


class TestBresenham:
    def test_vertical_line(self):
        """
        Test a vertical line from (0,0) to (0,10).
        Expects points progressing vertically with x constant.
        """

        start_point = Point(0, 0)
        end_point = Point(0, 10)

        ground_truth_points = [Point(0, i) for i in range(0, 11)]
        bresenham_points = Bresenham.compute_line(start_point, end_point)

        assert ground_truth_points == bresenham_points

    def test_horizontal_line(self):
        """
        Test a horizontal line from (0,0) to (10,0).
        Expects points progressing horizontally with y constant.
        """

        start_point = Point(0, 0)
        end_point = Point(10, 0)

        ground_truth_points = [Point(i, 0) for i in range(0, 11)]
        bresenham_points = Bresenham.compute_line(start_point, end_point)

        assert ground_truth_points == bresenham_points

    def test_diagonal_line(self):
        """
        Test a diagonal line with a slope of 1 from (0,0) to (10,10).
        Expects equal increments in x and y for a 45-degree line.
        """

        start_point = Point(0, 0)
        end_point = Point(10, 10)

        ground_truth_points = [Point(i, i) for i in range(0, 11)]
        bresenham_points = Bresenham.compute_line(start_point, end_point)

        assert ground_truth_points == bresenham_points

    def test_shallow_line(self):
        """
        Test a shallow line from (0,0) to (5,2).
        Expects fewer y changes due to shallow slope.
        """

        start_point = Point(0, 0)
        end_point = Point(5, 2)

        ground_truth_points = [Point(0, 0), Point(1, 0), Point(2, 1), Point(3, 1), Point(4, 2), Point(5, 2)]
        bresenham_points = Bresenham.compute_line(start_point, end_point)

        assert ground_truth_points == bresenham_points

    def test_steep_line(self):
        """
        Test a steep line from (0,0) to (2,5).
        Expects more frequent y-axis steps due to steep slope.
        """

        start_point = Point(0, 0)
        end_point = Point(2, 5)

        ground_truth_points = [Point(0, 0), Point(0, 1), Point(1, 2), Point(1, 3), Point(2, 4), Point(2, 5)]
        bresenham_points = Bresenham.compute_line(start_point, end_point)

        assert ground_truth_points == bresenham_points

    def test_negative_slope_two(self):
        """
        Test a line with a negative slope of -1 from (0,0) to (-10,10).
        Expects points where x decreases as y increases.
        """

        start_point = Point(0, 0)
        end_point = Point(-10, 10)

        ground_truth_points = [Point(-i, i) for i in range(10, -1, -1)]
        bresenham_points = Bresenham.compute_line(start_point, end_point)

        assert ground_truth_points == bresenham_points

    def test_negative_slope_three(self):
        """
        Test a line with a negative slope of -1 from (0,0) to (-10,-10).
        Expects points where both x and y decrease together.
        """

        start_point = Point(0, 0)
        end_point = Point(-10, -10)

        ground_truth_points = [Point(-i, -i) for i in range(10, -1, -1)]
        bresenham_points = Bresenham.compute_line(start_point, end_point)

        assert ground_truth_points == bresenham_points

    def test_negative_slope_four(self):
        """
        Test a line with a negative slope of 1 from (0,0) to (10,-10).
        Expects points where x increases and y decreases at the same rate.
        """

        start_point = Point(0, 0)
        end_point = Point(10, -10)

        ground_truth_points = [Point(i, -i) for i in range(0, 11)]
        bresenham_points = Bresenham.compute_line(start_point, end_point)

        assert ground_truth_points == bresenham_points

    def test_single_point(self):
        """
        Test a single-point line from (0,0) to (0,0).
        Expects a single point as output.
        """

        start_point = Point(0, 0)
        end_point = Point(0, 0)

        ground_truth_points = [Point(0, 0)]
        bresenham_points = Bresenham.compute_line(start_point, end_point)

        assert ground_truth_points == bresenham_points

    def test_off_by_one(self):
        """
        Test a slightly sloped line from (0,0) to (10,1).
        Expects x increment with minimal y steps due to shallow slope.
        """

        start_point = Point(0, 0)
        end_point = Point(10, 1)

        ground_truth_points = [
            Point(0, 0),
            Point(1, 0),
            Point(2, 0),
            Point(3, 0),
            Point(4, 0),
            Point(5, 1),
            Point(6, 1),
            Point(7, 1),
            Point(8, 1),
            Point(9, 1),
            Point(10, 1),
        ]
        bresenham_points = Bresenham.compute_line(start_point, end_point)

        assert ground_truth_points == bresenham_points
