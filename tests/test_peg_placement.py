from stringart.line_algorithms.matrix import MatrixGenerator


class TestPegPlacement:
    """Test class to verify peg placement methods for different shape configurations.
    The tests are based on different peg placement modes, including 'center' (square image),
    'first-half' (rectangle), and 'second-half' (rectangle).
    """

    shapes = [(20, 20), (20, 40), (40, 20)]

    # fmt: off
    def test_center_placement(self):
        pegs_ground_truth = [
            # (20, 20)
            [[9, 19], [14, 17], [17, 12], [17, 7], [14, 2], [9, 1], [3, 2], [0, 7], [0, 12], [3, 17]],
            # (20, 40)
            [[9, 29], [14, 27], [17, 22], [17, 17], [14, 12], [9, 11], [3, 12], [0, 17], [0, 22], [3, 27]],
            # (40, 20)
            [[20, 18], [25, 16], [28, 11], [28, 6], [25, 1], [20, 0], [14, 1], [11, 6], [11, 11], [14, 16]]
        ]
        pegs_computed = []

        for shape in TestPegPlacement.shapes:
            _, pegs = MatrixGenerator.compute_matrix(shape, 10, image_mode="center")
            pegs_arr = [[point.y, point.x] for point in pegs]

            pegs_computed.append(pegs_arr)
            
        assert pegs_computed == pegs_ground_truth


    def test_first_half_placement(self):
        # (20, 20), (20, 40), (40, 20)
        pegs_ground_truth = [[9, 18], [14, 16], [17, 11], [17, 6], [14, 1], [9, 0], [3, 1], [0, 6], [0, 11], [3, 16]]
        pegs_computed = []

        for shape in TestPegPlacement.shapes:
            _, pegs = MatrixGenerator.compute_matrix(shape, 10, image_mode="first-half")
            pegs_arr = [[point.y, point.x] for point in pegs]

            pegs_computed.append(pegs_arr)

        assert pegs_computed[0] == pegs_computed[1] == pegs_computed[2] == pegs_ground_truth

    def test_second_half_placement(self):
        pegs_ground_truth = [
            # (20, 20)
            [[9, 20], [14, 18], [17, 13], [17, 8], [14, 3], [9, 2], [3, 3], [0, 8], [0, 13], [3, 18]],
            # (20, 40)
            [[9, 40], [14, 38], [17, 33], [17, 28], [14, 23], [9, 22], [3, 23], [0, 28], [0, 33], [3, 38]],
            # (40, 20)
            [[31, 18], [36, 16], [39, 11], [39, 6], [36, 1], [31, 0], [25, 1], [22, 6], [22, 11], [25, 16]]
        ]
        pegs_computed = []

        for shape in TestPegPlacement.shapes:
            _, pegs = MatrixGenerator.compute_matrix(shape, 10, image_mode="second-half")
            pegs_arr = [[point.y, point.x] for point in pegs]

            pegs_computed.append(pegs_arr)

        assert pegs_computed == pegs_ground_truth
    # fmt: on
