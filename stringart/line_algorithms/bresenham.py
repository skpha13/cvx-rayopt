from collections import namedtuple
from typing import List

Point = namedtuple("Point", ["x", "y"])


class Bresenham:
    @staticmethod
    def compute_line(point0: Point, point1: Point):
        """Computes a list of points on a line between two endpoints using Bresenham's algorithm.

        Parameters
        ----------
        point0 : Point
            The starting point of the line.
        point1 : Point
            The ending point of the line.

        Returns
        -------
        List[Point]
            A list of points representing the line from `point0` to `point1`.
        """

        if abs(point0.x - point1.x) >= abs(point0.y - point1.y):
            if point0.x > point1.x:
                return Bresenham.draw_horizontal_line(point1, point0)

            return Bresenham.draw_horizontal_line(point0, point1)

        if point0.y > point1.y:
            return Bresenham.draw_vertical_line(point1, point0)

        return Bresenham.draw_vertical_line(point0, point1)

    @staticmethod
    def draw_horizontal_line(point0: Point, point1: Point) -> List[Point]:
        """Draws a line between two points, primarily moving horizontally (along the x-axis),
        using Bresenham's line drawing algorithm.

        Parameters
        ----------
        point0 : Point
            The starting point of the line.
        point1 : Point
            The ending point of the line.

        Returns
        -------
        List[Point]
            A list of points representing the line from `point0` to `point1`.
        """
        points: List[Point] = []
        direction = -1 if (point1.y - point0.y) < 0 else 1

        dx = point1.x - point0.x
        dy = (point1.y - point0.y) * direction
        y = point0.y
        D = 2 * dy - dx

        for i in range(dx + 1):
            points.append(Point(point0.x + i, y))

            if D >= 0:
                y += direction
                D -= 2 * dx

            D += 2 * dy

        return points

    @staticmethod
    def draw_vertical_line(point0: Point, point1: Point) -> List[Point]:
        """Draws a line between two points, primarily moving vertically (along the y-axis),
        using Bresenham's line drawing algorithm.

        Parameters
        ----------
        point0 : Point
            The starting point of the line.
        point1 : Point
            The ending point of the line.

        Returns
        -------
        List[Point]
            A list of points representing the line from `point0` to `point1`.
        """
        points: List[Point] = []
        direction = -1 if (point1.x - point0.x) < 0 else 1

        dx = (point1.x - point0.x) * direction
        dy = point1.y - point0.y
        x = point0.x
        D = 2 * dx - dy

        for i in range(dy + 1):
            points.append(Point(x, point0.y + i))

            if D >= 0:
                x += direction
                D -= 2 * dy

            D += 2 * dx

        return points

    @staticmethod  # TODO: maybe replace default values
    def compute_matrix(number_of_pegs: int = 200, radius: int | None = None):
        pass
