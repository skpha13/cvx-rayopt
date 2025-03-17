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
            solver="least-squares",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            matrix_representation="dense",
            mp_method=None,
            number_of_lines=10,
            selector_type=None,
        )

        configuration.run_configuration(running_tests=True)

    def test_lsq_sparse(self):
        configuration = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="least-squares",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            matrix_representation="sparse",
            mp_method=None,
            number_of_lines=10,
            selector_type=None,
        )

        configuration.run_configuration(running_tests=True)

    def test_gmp_random(self):
        configuration = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="matching-pursuit",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            matrix_representation=None,
            mp_method="greedy",
            number_of_lines=10,
            selector_type="random",
        )

        configuration.run_configuration(running_tests=True)

    def test_gmp_dotproduct(self):
        configuration = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="matching-pursuit",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            matrix_representation=None,
            mp_method="greedy",
            number_of_lines=10,
            selector_type="dot-product",
        )

        configuration.run_configuration(running_tests=True)

    def test_omp(self):
        configuration = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="matching-pursuit",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            matrix_representation=None,
            mp_method="orthogonal",
            number_of_lines=10,
            selector_type=None,
        )

        configuration.run_configuration(running_tests=True)

    def test_crop_modes(self):
        configuration_first_half = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="least-squares",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="first-half",
            matrix_representation="dense",
            mp_method=None,
            number_of_lines=10,
            selector_type=None,
        )

        configuration_second_half = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="least-squares",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="second-half",
            matrix_representation="dense",
            mp_method=None,
            number_of_lines=10,
            selector_type=None,
        )

        configuration_center = Configuration(
            metadata=self.metadata,
            command="solve",
            solver="least-squares",
            image_path=self.img_path,
            number_of_pegs=10,
            crop_mode="center",
            matrix_representation="dense",
            mp_method=None,
            number_of_lines=10,
            selector_type=None,
        )

        configuration_first_half.run_configuration(running_tests=True)
        configuration_second_half.run_configuration(running_tests=True)
        configuration_center.run_configuration(running_tests=True)
