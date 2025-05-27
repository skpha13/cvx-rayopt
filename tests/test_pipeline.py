import os
from pathlib import Path

from stringart.cli_functions import Configuration
from stringart.utils.types import Metadata


class TestPipeline:
    @classmethod
    def setup_class(cls):
        stringart_directory: Path = Path(os.path.dirname(os.path.abspath(__file__)))
        directory: Path = stringart_directory.parent.resolve()
        metadata = Metadata(directory)

        cls.img_path = metadata.path / "imgs" / "lena.png"
        cls.metadata = metadata

    def test_lsq_dense(self):
        configuration = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="ls",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            rasterization="bresenham",
            matrix_representation="dense",
        )
        configuration.run_configuration(running_tests=True)

        configuration = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="ls",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            rasterization="bresenham",
            matrix_representation="dense",
            number_of_lines=10,
            binary=True,
        )

        configuration.run_configuration(running_tests=True)

        configuration = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="ls",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            rasterization="bresenham",
            matrix_representation="dense",
            number_of_lines=10,
            block_size=2,
        )

        configuration.run_configuration(running_tests=True)

    def test_lsq_sparse(self):
        configuration = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="ls",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            rasterization="xiaolin-wu",
            matrix_representation="sparse",
            number_of_lines=10,
        )

        configuration.run_configuration(running_tests=True)

    def test_lsq_linear(self):
        configuration = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="lls",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            rasterization="xiaolin-wu",
            matrix_representation="sparse",
            number_of_lines=10,
        )

        configuration.run_configuration(running_tests=True)

    def test_gmp_random(self):
        configuration = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="mp",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            mp_method="greedy",
            number_of_lines=10,
            selector_type="random",
            block_size=2,
        )

        configuration.run_configuration(running_tests=True)

    def test_gmp_dotproduct(self):
        configuration = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="mp",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            rasterization="bresenham",
            mp_method="greedy",
            number_of_lines=10,
            selector_type="dot-product",
            block_size=2,
        )

        configuration.run_configuration(running_tests=True)

    def test_gmp_all(self):
        configuration = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="mp",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            rasterization="bresenham",
            mp_method="greedy",
            number_of_lines=10,
            selector_type="all",
            block_size=2,
        )

        configuration.run_configuration(running_tests=True)

    def test_omp(self):
        configuration = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="mp",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            rasterization="xiaolin-wu",
            mp_method="orthogonal",
            number_of_lines=10,
            block_size=2,
        )

        configuration.run_configuration(running_tests=True)

    def test_crop_modes(self):
        configuration_first_half = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="ls",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="first-half",
            matrix_representation="dense",
            number_of_lines=10,
        )

        configuration_second_half = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="ls",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="second-half",
            matrix_representation="dense",
            rasterization="bresenham",
            number_of_lines=10,
        )

        configuration_center = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="ls",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            matrix_representation="dense",
            number_of_lines=10,
        )

        configuration_first_half.run_configuration(running_tests=True)
        configuration_second_half.run_configuration(running_tests=True)
        configuration_center.run_configuration(running_tests=True)

    def test_binary_projection_ls(self):
        configuration_defaults = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="bpls",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            rasterization="xiaolin-wu",
            number_of_lines=10,
            block_size=2,
        )

        configuration_cvxopt = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="bpls",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            rasterization="xiaolin-wu",
            matrix_representation="sparse",
            number_of_lines=10,
            qp_solver="cvxopt",
            k=3,
            max_iterations=1,
            block_size=2,
        )

        configuration_scipy = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="bpls",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            rasterization="xiaolin-wu",
            matrix_representation="dense",
            number_of_lines=10,
            qp_solver="scipy",
            k=10,
            max_iterations=5,
            block_size=2,
        )

        configuration_defaults.run_configuration(running_tests=True)
        configuration_cvxopt.run_configuration(running_tests=True)
        configuration_scipy.run_configuration(running_tests=True)

    def test_least_squares_regularized(self):
        configuration_no_reg = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="lsr",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            rasterization="xiaolin-wu",
        )

        configuration_smooth = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="lsr",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            rasterization="xiaolin-wu",
            regularizer="smooth",
            lambd=0.1,
        )

        configuration_abs = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="lsr",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            rasterization="xiaolin-wu",
            regularizer="abs",
            lambd=0.1,
        )

        configuration_no_reg.run_configuration(running_tests=True)
        configuration_smooth.run_configuration(running_tests=True)
        configuration_abs.run_configuration(running_tests=True)
