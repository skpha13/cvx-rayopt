import math
from typing import List, Tuple

import matplotlib.pyplot as plt
from stringart.utils.types import Point


class XiaolinWu:
    """A class that implements the Xiaolin Wu's antialiased line algorithm.

    This algorithm computes the pixel positions and associated intensities for a line
    between two points in a way that smooths the line's appearance by antialiasing.
    """

    @staticmethod
    def compute_line(point0: Point, point1: Point) -> Tuple[List[Point], List[float]]:
        """Computes the anti-aliased line between two points.

        This method uses the Xiaolin Wu algorithm to determine which pixels to
        draw and their corresponding intensity values for a line between the given
        points `point0` and `point1`.

        Parameters
        ----------
        point0 : Point
            The starting point of the line.
        point1 : Point
            The ending point of the line.

        Returns
        -------
        indices : List[Point]
            A list of points that represent the pixels along the line.
        values : List[float]
            A list of intensity values corresponding to each pixel.
        """

        indices = []
        values = []

        dx = point1.x - point0.x
        dy = point1.y - point0.y

        if dx == 0:  # handle vertical line
            for y in range(min(point0.y, point1.y), max(point0.y, point1.y) + 1):
                indices.append(Point(point0.x, y))  # x is constant for vertical lines
                values.append(1.0)

            return indices, values

        elif dy == 0:  # handle horizontal line
            for x in range(min(point0.x, point1.x), max(point0.x, point1.x) + 1):
                indices.append(Point(x, point0.y))  # y is constant for horizontal lines
                values.append(1.0)

            return indices, values

        steep = abs(point1.y - point0.y) > abs(point1.x - point0.x)
        if steep:
            point0 = Point(point0.y, point0.x)
            point1 = Point(point1.y, point1.x)
        if point0.x > point1.x:
            point0, point1 = point1, point0

        dx = point1.x - point0.x
        dy = point1.y - point0.y
        gradient = dy / dx if dx != 0 else 1

        # handle first endpoint
        idx, val = XiaolinWu.handle_endpoint(point0, gradient, steep, True)
        interpolated_y = point0.y + gradient * (round(point0.x) - point0.x) + gradient
        indices += idx
        values += val

        # handle the second endpoint
        idx, val = XiaolinWu.handle_endpoint(point1, gradient, steep, False)
        indices += idx
        values += val

        # main loop
        xpx1 = round(point0.x)
        xpx2 = round(point1.x)
        if steep:
            for x in range(xpx1 + 1, xpx2):
                indices.append(Point(XiaolinWu.ipart(interpolated_y), x))
                values.append(XiaolinWu.rfpart(interpolated_y))

                indices.append(Point(XiaolinWu.ipart(interpolated_y) + 1, x))
                values.append(XiaolinWu.fpart(interpolated_y))

                interpolated_y += gradient
        else:
            for x in range(xpx1 + 1, xpx2):
                indices.append(Point(x, XiaolinWu.ipart(interpolated_y)))
                values.append(XiaolinWu.rfpart(interpolated_y))

                indices.append(Point(x, XiaolinWu.ipart(interpolated_y) + 1))
                values.append(XiaolinWu.fpart(interpolated_y))

                interpolated_y += gradient

        return indices, values

    @staticmethod
    def handle_endpoint(point: Point, gradient: float, steep: bool, is_first: bool) -> Tuple[List[Point], List[float]]:
        """Handles the calculation of pixel positions and intensities for an endpoint.

        This method determines the appropriate intensity values and pixel positions
        for the first and last pixel of the line based on the line's gradient.

        Parameters
        ----------
        point : Point
            The point at either the start or end of the line.
        gradient : float
            The gradient (slope) of the line.
        steep : bool
            A flag indicating whether the line is steep (i.e., vertical or near-vertical).
        is_first : bool
            A flag indicating whether the point is the first endpoint (True) or second endpoint (False).

        Returns
        -------
        indices : List[Point]
            A list of points representing the endpoint pixel(s).
        values : List[float]
            A list of intensity values for the endpoint(s).
        """

        indices = []
        values = []

        x_end = round(point.x)
        y_end = point.y + gradient * (x_end - point.x)

        x_gap = XiaolinWu.rfpart(point.x + 0.5) if is_first else XiaolinWu.fpart(point.x + 0.5)
        x_px = x_end
        y_px = XiaolinWu.ipart(y_end)

        if steep:
            indices.append(Point(y_px, x_px))
            values.append(XiaolinWu.rfpart(y_end) * x_gap)

            indices.append(Point(y_px, x_px))
            values.append(XiaolinWu.fpart(y_end) * x_gap)
        else:
            indices.append(Point(x_px, y_px))
            values.append(XiaolinWu.rfpart(y_end) * x_gap)

            indices.append(Point(x_px, y_px))
            values.append(XiaolinWu.fpart(y_end) * x_gap)

        return indices, values

    @staticmethod
    def ipart(x):
        """Returns the integer part of a floating-point value."""
        return int(math.floor(x))

    @staticmethod
    def fpart(x):
        """Returns the fractional part of a floating-point value."""
        return x - math.floor(x)

    @staticmethod
    def rfpart(x):
        """Returns the reverse fractional part of a floating-point value."""
        return 1 - XiaolinWu.fpart(x)


if __name__ == "__main__":
    # example usage
    p0 = Point(20, 30)
    p1 = Point(180, 150)
    indices, values = XiaolinWu.compute_line(p0, p1)

    x_coords = [pt.x for pt in indices]
    y_coords = [pt.y for pt in indices]

    plt.scatter(x_coords, y_coords, c=values, cmap="hot", marker="o", label="Line Pixels", s=100)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Xiaolin Wu Line Algorithm")

    plt.xlim(20, 50)
    plt.ylim(30, 50)

    plt.legend()
    plt.grid(True)
    plt.show()
