# Procedural-Computing-of-String-Art

StringArt is a Python-based package used to generate string art configurations from an input image.

## User Documentation

### Installation    

To get started, install the package by following these steps:

```bash
# clone repository
git clone https://github.com/skpha13/Procedural-Computing-of-String-Art.git

# enter the directory 
cd Procedural-Computing-of-String-Art

# install all required dependencies
pip install .
```

### Command Line Interface (CLI)

This package provides a simple and intuitive CLI for computing string art images.

#### Summary of Key Commands 

| **Command**         | **Description**                                                                                                                  |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `run-benchmarks`    | Run all benchmarks for StringArt.                                                                                                |
| `run-analysis`      | Run analysis on StringArt benchmarks.                                                                                            |
| `solve`             | Compute StringArt configurations using the specified solver and options.                                                         |
| `--help`            | Displays help information.                                                                                                       |

#### Common Arguments

| **Argument**       | **Description**                                                                                                        |
|--------------------|------------------------------------------------------------------------------------------------------------------------|
| `--image-path`     | Path to the input image. Required.                                                                                     |
| `--number-of-pegs` | Number of pegs to use. Optional. Defaults to 100 if not specified.                                                     |
| `--crop-mode`      | Crop mode to apply to the input image. Choices: `first-half`, `center`, `second-half`. Optional. Defaults to `center`. |
| `--rasterization`  | Line rasterization algorithm. Choices: `bresenham` (fast integer-based), `xiaolin-wu` (anti-aliased). Optional.        |

#### Solves

| **Solvers**                 | **Description**                                                                                                                                    |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `least-squares`             | Solves the system using a standard least squares method. Suitable for dense or sparse matrix formulations.                                         |
| `linear-least-squares`      | Alias or variant of least-squares solver focusing on linear positive constraints.                                                                  |
| `binary-projection-ls`      | Solves a binary-constrained problem by iteratively projecting least squares solutions to binary values.                                            |
| `least-squares-regularized` | Solves a regularized least squares problem using quadratic programming.                                                                            |
| `matching-pursuit`          | A greedy method that incrementally builds a solution by selecting atoms (lines) based on correlation. Supports `greedy` and `orthogonal` variants. |


#### Least Squares Solver Arguments

| **Argument**              | **Description**                                                                                                         |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `--matrix-representation` | Specify the matrix representation method for the `least-squares` solver. Choices: `dense`, `sparse`. Default: `sparse`. |
| `--number-of-lines`       | Optional. Number of top lines to select.                                                                                |

#### Binary Projection LS Solver Arguments

| **Argument**              | **Description**                                                                                                      |
|---------------------------|----------------------------------------------------------------------------------------------------------------------|
| `--qp-solver`             | Quadratic programming solver to use for least squares step. Choices: `cvxopt`, `scipy`. Optional. Default: `cvxopt`. |
| `--matrix-representation` | Matrix representation method to use. Choices: `dense`, `sparse`. Optional. Default: `sparse`.                        |
| `--k`                     | Number of variables to fix to 1 in each iteration. Optional. Default: `3`                                            |
| `--max-iterations`        | Maximum number of iterations to run before stopping. Optional. Default: `100`                                        |

#### Least Squares Regularized Solver Arguments

| **Argument**              | **Description**                                                                               |
|---------------------------|-----------------------------------------------------------------------------------------------|
| `--matrix-representation` | Matrix representation method to use. Choices: `dense`, `sparse`. Optional. Default: `sparse`. |
| `--regularizer`           | The type of regularization to apply. Choices: `None`, `smooth`, `abs`. Default is `None`.     |
| `--lambda`                | The regularization strength. Defaults to `0.1`.                                               |

#### Matching Pursuit Solver Arguments

| **Argument**        | **Description**                                                                                                  |
|---------------------|------------------------------------------------------------------------------------------------------------------|
| `--method`          | Algorithm selection for matching pursuit. Choices: `greedy`, `orthogonal`. Optional. Default: `orthogonal`.      |
| `--selector`        | Selector method to use with `greedy` method. Choices: `random`, `dot-product`. Optional. Default: `dot-product`. |
| `--number-of-lines` | Required. Number of top lines to select.                                                                         |


#### Example Commands:

```bash
# run all benchmarks and save output to the benchmarks folder
python ./stringart/main.py run-benchmarks --image-path ./imgs/lena.png 

# run analysis on provided benchmarks
# this should be run after the `run-benchmarks` command.
python ./stringart/main.py run-analysis --image-path ./imgs/lena.png 

# runs the least squares solver with the sparse matrix representation on the provided image. The number of pegs used will be 100, the crop mode for the image center and the rasterization algorithm xiaolin-wu.
python ./stringart/main.py solve least-squares --image-path ./imgs/lena.png --rasterization xiaolin-wu 

# runs the matching pursuit solver with the orthogonal method (OMP) on the provided image, selecting 1000 lines.
python ./stringart/main.py solve matching-pursuit --image-path ./imgs/lena.png --number-of-lines 1000 --method orthogonal 

# runs the matching pursuit solver with the greedy method on the provided image, using the dot-product heuristic, selecting 1000 lines.
python ./stringart/main.py solve matching-pursuit --image-path ./imgs/lena.png --number-of-lines 1000 --method greedy

# runs the least squares solver with the sparse matrix representation, a crop mode using the first half of the image and a number of pegs of 50
python ./stringart/main.py solve least-squares  --image-path ./imgs/lena.png --crop-mode first-half --number-of-pegs 50 

# runs the linear least squares solver with a selection of 1000 lines
python ./stringart/main.py solve linear-least-squares --number-of-lines 1000 --image-path ./imgs/lena.png --rasterization xiaolin-wu

# runs the binary projection least squares with the `scipy` solver
 python ./stringart/main.py solve binary-projection-ls --qp-solver scipy --k 500 --max-iterations 1 --image-path ./imgs/lena.png
 
# runs the regularized least squares with the `smooth` regularizer and a strength of 10.
 python ./stringart/main.py solve least-squares-regularized --regularizer "smooth" --lambda 10 --image-path ./imgs/lena.png --rasterization xiaolin-wu
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