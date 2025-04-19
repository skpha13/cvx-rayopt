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
            [[10, 18], [14, 16], [17, 12], [17, 7], [14, 3], [10, 2], [5, 3], [2, 7], [2, 12], [5, 16]],
            # (20, 40)
            [[10, 28], [14, 26], [17, 22], [17, 17], [14, 13], [10, 12], [5, 13], [2, 17], [2, 22], [5, 26]],
            # (40, 20)
            [[20, 18], [24, 16], [27, 12], [27, 7], [24, 3], [20, 2], [15, 3], [12, 7], [12, 12], [15, 16]]
        ]
        pegs_computed = []

        for shape in TestPegPlacement.shapes:
            _, pegs = MatrixGenerator.compute_matrix(shape, 10, crop_mode="center")
            pegs_arr = [[point.y, point.x] for point in pegs]

            pegs_computed.append(pegs_arr)
            
        assert pegs_computed == pegs_ground_truth


    def test_first_half_placement(self):
        # (20, 20), (20, 40), (40, 20)
        pegs_ground_truth = [[10, 18], [14, 16], [17, 12], [17, 7], [14, 3], [10, 2], [5, 3], [2, 7], [2, 12], [5, 16]]
        pegs_computed = []

        for shape in TestPegPlacement.shapes:
            _, pegs = MatrixGenerator.compute_matrix(shape, 10, crop_mode="first-half")
            pegs_arr = [[point.y, point.x] for point in pegs]

            pegs_computed.append(pegs_arr)

        assert pegs_computed[0] == pegs_computed[1] == pegs_computed[2] == pegs_ground_truth

    def test_second_half_placement(self):
        pegs_ground_truth = [
            # (20, 20)
            [[10, 18], [14, 16], [17, 12], [17, 7], [14, 3], [10, 2], [5, 3], [2, 7], [2, 12], [5, 16]],
            # (20, 40)
            [[10, 38], [14, 36], [17, 32], [17, 27], [14, 23], [10, 22], [5, 23], [2, 27], [2, 32], [5, 36]],
            # (40, 20)
            [[30, 18], [34, 16], [37, 12], [37, 7], [34, 3], [30, 2], [25, 3], [22, 7], [22, 12], [25, 16]]
        ]
        pegs_computed = []

        for shape in TestPegPlacement.shapes:
            _, pegs = MatrixGenerator.compute_matrix(shape, 10, crop_mode="second-half")
            pegs_arr = [[point.y, point.x] for point in pegs]

            pegs_computed.append(pegs_arr)

        assert pegs_computed == pegs_ground_truth
    # fmt: on
