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

[//]: # (TODO: add rasterization)

| **Command**        | **Description**                                                                                                                   |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `run-benchmarks`   | Run all benchmarks for StringArt.                                                                                                 |
| `run-analysis`     | Run analysis on StringArt benchmarks.                                                                                             |
| `solve`            | Compute StringArt configurations using the specified solver and options.                                                          |
| `--help`           | Displays help information.                                                                                                        |
| `--image-path`     | File path to the image to process. Supported formats: PNG, JPEG. Required.                                                        |
| `--number-of-pegs` | Number of pegs to use in computation. Default: 100.                                                                               |
| `--crop-mode`      | Specify the crop mode for the image. Choices: `first-half`, `center`, `second-half`. Default: `center`.                           |
| `--rasterization`  | Specify the rasterization algorithm to use for drawing the StringArt. Choices: `bresenham` or `xiaolin-wu`. Default: `xiaolin-wu` |

#### Solve Arguments

| **Argument**              | **Description**                                                                                    |
|---------------------------|----------------------------------------------------------------------------------------------------|
| `--solver`                | Specify the solver to use for computation. Choices: `least-squares`, `matching-pursuit`. Required. |


#### Least Squares Solver Arguments

| **Argument**              | **Description**                                                                                                         |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `--matrix-representation` | Specify the matrix representation method for the `least-squares` solver. Choices: `dense`, `sparse`. Default: `sparse`. |

#### Matching Pursuit Solver Arguments

| **Argument**              | **Description**                                                                                                                         |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `--method`                | Specify the algorithm selection for `matching-pursuit`. Choices: `greedy`, `orthogonal`.                                                |
| `--number-of-lines`       | Specify the number of lines to select for `matching-pursuit`.                                                                           |
| `--selector`              | Specify the selector method for `matching-pursuit` with the `greedy` method. Choices: `random`, `dot-product`. Default: `dot-product`.  |

#### Example Commands:

```bash
# run all benchmarks and save output to the benchmarks folder
python ./stringart/main.py --image-path ./imgs/lena.png run-benchmarks

# run analysis on provided benchmarks
# this should be run after the `run-benchmarks` command.
python ./stringart/main.py --image-path ./imgs/lena.png run-analysis

# runs the least squares solver with the sparse matrix representation on the provided image. The number of pegs used will be 100, the crop mode for the image center and the rasterization algorithm xiaolin-wu.
python ./stringart/main.py --image-path ./imgs/lena.png --rasterization xiaolin-wu solve --solver least-squares 

# runs the matching pursuit solver with the orthogonal method (OMP) on the provided image, selecting 1000 lines.
python ./stringart/main.py --image-path ./imgs/lena.png solve --solver matching-pursuit --method orthogonal --number-of-lines 1000

# runs the matching pursuit solver with the greedy method on the provided image, using the dot-product heuristic, selecting 1000 lines.
python ./stringart/main.py --image-path ./imgs/lena.png solve --solver matching-pursuit --method greedy --number-of-lines 1000

# runs the least squares solver with the sparse matrix representation, a crop mode using the first half of the image and a number of pegs of 50
python ./stringart/main.py --image-path ./imgs/lena.png --crop-mode first-half --number-of-pegs 50 solve --solver least-squares 
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