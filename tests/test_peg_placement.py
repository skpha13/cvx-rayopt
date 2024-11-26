from stringart.line_algorithms.matrix import DenseMatrixGenerator


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
            [[10, 19], [15, 17], [18, 12], [18, 7], [15, 2], [10, 1], [4, 2], [1, 7], [1, 12], [4, 17]],
            # (20, 40)
            [[10, 29], [15, 27], [18, 22], [18, 17], [15, 12], [10, 11], [4, 12], [1, 17], [1, 22], [4, 27]],
            # (40, 20)
            [[20, 19], [25, 17], [28, 12], [28, 7], [25, 2], [20, 1], [14, 2], [11, 7], [11, 12], [14, 17]]
        ]
        pegs_computed = []

        for shape in TestPegPlacement.shapes:
            _, pegs, _ = DenseMatrixGenerator.compute_matrix(shape, 10, mode="center")
            pegs_arr = [[point.y, point.x] for point in pegs]

            pegs_computed.append(pegs_arr)
            
        assert pegs_computed == pegs_ground_truth


    def test_first_half_placement(self):
        # (20, 20), (20, 40), (40, 20)
        pegs_ground_truth = [[10, 19], [15, 17], [18, 12], [18, 7], [15, 2], [10, 1], [4, 2], [1, 7], [1, 12], [4, 17]]
        pegs_computed = []

        for shape in TestPegPlacement.shapes:
            _, pegs, _ = DenseMatrixGenerator.compute_matrix(shape, 10, mode="first-half")
            pegs_arr = [[point.y, point.x] for point in pegs]

            pegs_computed.append(pegs_arr)

        assert pegs_computed[0] == pegs_computed[1] == pegs_computed[2] == pegs_ground_truth

    def test_second_half_placement(self):
        pegs_ground_truth = [
            # (20, 20)
            [[10, 19], [15, 17], [18, 12], [18, 7], [15, 2], [10, 1], [4, 2], [1, 7], [1, 12], [4, 17]],
            # (20, 40)
            [[10, 39], [15, 37], [18, 32], [18, 27], [15, 22], [10, 21], [4, 22], [1, 27], [1, 32], [4, 37]],
            # (40, 20)
            [[30, 19], [35, 17], [38, 12], [38, 7], [35, 2], [30, 1], [24, 2], [21, 7], [21, 12], [24, 17]]
        ]
        pegs_computed = []

        for shape in TestPegPlacement.shapes:
            _, pegs, _ = DenseMatrixGenerator.compute_matrix(shape, 10, mode="second-half")
            pegs_arr = [[point.y, point.x] for point in pegs]

            pegs_computed.append(pegs_arr)

        assert pegs_computed == pegs_ground_truth
    # fmt: on
