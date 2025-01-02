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
            [[10, 20], [15, 18], [19, 13], [19, 6], [15, 1], [10, 0], [4, 1], [0, 6], [0, 13], [4, 18]],
            # (20, 40)
            [[10, 30], [15, 28], [19, 23], [19, 16], [15, 11], [10, 10], [4, 11], [0, 16], [0, 23], [4, 28]],
            # (40, 20)
            [[20, 20], [25, 18], [29, 13], [29, 6], [25, 1], [20, 0], [14, 1], [10, 6], [10, 13], [14, 18]]
        ]
        pegs_computed = []

        for shape in TestPegPlacement.shapes:
            _, pegs, _ = MatrixGenerator.compute_matrix(shape, 10, mode="center")
            pegs_arr = [[point.y, point.x] for point in pegs]

            pegs_computed.append(pegs_arr)
            
        assert pegs_computed == pegs_ground_truth


    def test_first_half_placement(self):
        # (20, 20), (20, 40), (40, 20)
        pegs_ground_truth = [[10, 20], [15, 18], [19, 13], [19, 6], [15, 1], [10, 0], [4, 1], [0, 6], [0, 13], [4, 18]]
        pegs_computed = []

        for shape in TestPegPlacement.shapes:
            _, pegs, _ = MatrixGenerator.compute_matrix(shape, 10, mode="first-half")
            pegs_arr = [[point.y, point.x] for point in pegs]

            pegs_computed.append(pegs_arr)

        assert pegs_computed[0] == pegs_computed[1] == pegs_computed[2] == pegs_ground_truth

    def test_second_half_placement(self):
        pegs_ground_truth = [
            # (20, 20)
            [[10, 20], [15, 18], [19, 13], [19, 6], [15, 1], [10, 0], [4, 1], [0, 6], [0, 13], [4, 18]],
            # (20, 40)
            [[10, 40], [15, 38], [19, 33], [19, 26], [15, 21], [10, 20], [4, 21], [0, 26], [0, 33], [4, 38]],
            # (40, 20)
            [[30, 20], [35, 18], [39, 13], [39, 6], [35, 1], [30, 0], [24, 1], [20, 6], [20, 13], [24, 18]]
        ]
        pegs_computed = []

        for shape in TestPegPlacement.shapes:
            _, pegs, _ = MatrixGenerator.compute_matrix(shape, 10, mode="second-half")
            pegs_arr = [[point.y, point.x] for point in pegs]

            pegs_computed.append(pegs_arr)

        assert pegs_computed == pegs_ground_truth
    # fmt: on
