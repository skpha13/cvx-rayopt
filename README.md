# Convex Optimization Relaxation for Radial Image Reconstruction

StringArt is a Python package designed to generate string art configurations based on an input image, leveraging convex optimization techniques for radial image reconstruction.

## Overview

For a comprehensive understanding of the mathematical formulation and approach, please refer to the thesis paper:

[**Convex Optimization Relaxation for Radial Image Reconstruction**](./documentation.pdf)

For a concise summary, usage instructions, and additional code insights, check the documentation folder:

[**docs**](./docs/README.md)

## User Documentation

### Installation    

To get started, install the package by following these steps:

```bash
# clone repository
git clone https://github.com/skpha13/cvx-rayopt.git

# enter the directory 
cd cvx-rayopt

# install all required dependencies
pip install .
```

### Command Line Interface (CLI)

This package provides a simple and intuitive CLI for computing string art images.

#### Summary of Key Commands 

| **Command**      | **Description**                                                          |
|------------------|--------------------------------------------------------------------------|
| `run-benchmarks` | Run all benchmarks for StringArt.                                        |
| `run-analysis`   | Run analysis on StringArt benchmarks.                                    |
| `solve`          | Compute StringArt configurations using the specified solver and options. |
| `--log-level`    | Set the logging level. Defaults to `INFO`.                               |
| `--help`         | Displays help information.                                               |

#### Common Arguments

| **Argument**       | **Description**                                                                                                                 |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `--image-path`     | Path to the input image. Required.                                                                                              |
| `--number-of-pegs` | Number of pegs to use. Default `128`.                                                                                           |
| `--crop-mode`      | Crop mode to apply to the input image. Choices: `first-half`, `center`, `second-half`. Defaults to `center`.                    |
| `--rasterization`  | Line rasterization algorithm. Choices: `bresenham` (fast integer-based), `xiaolin-wu` (anti-aliased). Defaults to `xiaolin-wu`. |
| `--block-size`     | Enables residual computation using supersampling. Example values: 2, 4, 8, 16. Defaults to `None`.                              |

#### Benchmark

| **Argument**   | **Description**                                                                           |
|----------------|-------------------------------------------------------------------------------------------|
| `--output-dir` | Name of the directory where benchmark results will be saved. Defaults to `benchmarks_01`. |

#### Analysis

| **Argument**            | **Description**                                                                        |
|-------------------------|----------------------------------------------------------------------------------------|
| `--input-benchmark-dir` | **Required.** Directory name containing the benchmark results to be analyzed.          |
| `--output-dir`          | Name of the directory where analysis results will be saved. Defaults to `analysis_01`. |


#### Solvers

| **Solvers** | **Description**                                                                                                                                    |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `ls`        | Solves the system using a standard least squares method. Suitable for dense or sparse matrix formulations.                                         |
| `lls`       | Alias or variant of least-squares solver focusing on linear positive constraints.                                                                  |
| `bpls`      | Solves a binary-constrained problem by iteratively projecting least squares solutions to binary values.                                            |
| `lsr`       | Solves a regularized least squares problem using quadratic programming.                                                                            |
| `mp`        | A greedy method that incrementally builds a solution by selecting atoms (lines) based on correlation. Supports `greedy` and `orthogonal` variants. |
| `radon`     | A greedy method that incrementally builds a solution by selecting atoms (lines) based on contributions to the Radon transform sinogram.            |


#### Least Squares Solver Arguments

| **Argument**              | **Description**                                                                                                            |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `--matrix-representation` | Specify the matrix representation method for the `least-squares` solver. Choices: `dense`, `sparse`. Defaults to `sparse`. |
| `--number-of-lines`       | Optional. Number of top lines to select.                                                                                   |
| `--binary`                | Optional. If enabled, projects `x` coefficients to binary values.                                                          |

#### Binary Projection LS Solver Arguments

| **Argument**              | **Description**                                                                                               |
|---------------------------|---------------------------------------------------------------------------------------------------------------|
| `--qp-solver`             | Quadratic programming solver to use for least squares step. Choices: `cvxopt`, `scipy`. Defaults to `cvxopt`. |
| `--matrix-representation` | Matrix representation method to use. Choices: `dense`, `sparse`. Defaults to `sparse`.                        |
| `--k`                     | Number of variables to fix to 1 in each iteration. Defaults to `3`                                            |
| `--max-iterations`        | Maximum number of iterations to run before stopping. Defaults to `100`                                        |
| `--lambda`                | The regularization strength. Defaults to `None`.                                                              |

#### Least Squares Regularized Solver Arguments

| **Argument**              | **Description**                                                                            |
|---------------------------|--------------------------------------------------------------------------------------------|
| `--matrix-representation` | Matrix representation method to use. Choices: `dense`, `sparse`. Defaults to `sparse`.     |
| `--regularizer`           | The type of regularization to apply. Choices: `None`, `smooth`, `abs`. Defaults to `None`. |
| `--lambda`                | The regularization strength. Defaults to `0.1`.                                            |

#### Matching Pursuit Solver Arguments

| **Argument**        | **Description**                                                                                           |
|---------------------|-----------------------------------------------------------------------------------------------------------|
| `--method`          | Algorithm selection for matching pursuit. Choices: `greedy`, `orthogonal`. Defaults to `orthogonal`.      |
| `--selector`        | Selector method to use with `greedy` method. Choices: `random`, `dot-product`. Defaults to `dot-product`. |
| `--number-of-lines` | Required. Number of top lines to select.                                                                  |


#### Example Commands:

```bash
# run all benchmarks and save output to the benchmarks folder
python ./stringart/main.py run-benchmarks --image-path ./imgs/lena.png 

# run analysis on provided benchmarks
# this should be run after the `run-benchmarks` command.
python ./stringart/main.py run-analysis --input-benchmark-dir benchmarks_01 --image-path ./imgs/lena.png 

# runs the least squares solver with the sparse matrix representation on the provided image. The number of pegs used will be 100, the crop mode for the image center and the rasterization algorithm xiaolin-wu.
python ./stringart/main.py solve ls --image-path ./imgs/lena.png --rasterization xiaolin-wu 

# runs the matching pursuit solver with the orthogonal method (OMP) on the provided image, selecting 1000 lines.
python ./stringart/main.py solve mp --image-path ./imgs/lena.png --number-of-lines 1000 --method orthogonal 

# runs the matching pursuit solver with the greedy method on the provided image, using the dot-product heuristic, selecting 1000 lines.
python ./stringart/main.py solve mp --image-path ./imgs/lena.png --number-of-lines 1000 --method greedy

# runs the least squares solver with the sparse matrix representation, a crop mode using the first half of the image and a number of pegs of 50
python ./stringart/main.py solve ls  --image-path ./imgs/lena.png --crop-mode first-half --number-of-pegs 50 

# runs the linear least squares solver with a selection of 1000 lines
python ./stringart/main.py solve lls --number-of-lines 1000 --image-path ./imgs/lena.png --rasterization xiaolin-wu

# runs the binary projection least squares with the `scipy` solver
 python ./stringart/main.py solve bpls --qp-solver scipy --k 500 --max-iterations 1 --image-path ./imgs/lena.png
 
# runs the regularized least squares with the `smooth` regularizer and a strength of 10.
 python ./stringart/main.py solve lsr --regularizer "smooth" --lambda 10 --image-path ./imgs/lena.png --rasterization xiaolin-wu
 
# runs the radon solver with a block_size of 8 for supersampling
python ./stringart/main.py solve radon --block-size 8 --number-of-pegs 256 --image-path ./imgs/lena.png 
```

## Developer Documentation

### Install Optional Packages

For development purposes, install the optional packages with:

```bash
pip install stringart[dev]
```

This will install the following tools:

- **Black:**  A code formatter to ensure consistent style.
- **isort:**  A tool for sorting imports automatically.
- **Pytest:** A testing framework for running unit tests.

### Running Tests

Run the test suite to ensure everything is working:

```bash
python -m pytest
```

## Observations

- The Python docstrings for functions and classes were generated with the assistance of **ChatGPT**. 
- Portions of the **README**, particularly repetitive sections such as the command table, were also created with the help of **ChatGPT**.