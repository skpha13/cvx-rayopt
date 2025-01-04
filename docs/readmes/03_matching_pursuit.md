# Matching Pursuit

## Terminology

**"Matching Pursuit"** is a specific greedy algorithm applied in signal processing and machine learning to approximate a signal using a linear combination of *dictionary atoms* (columns from the matrix)

It iteratively selects the *atom* that at the current step reduces the residual most significantly.

This concept has been adopted and integrated into this context to explore its performance and behaviour.

You can read more here: [Wikipedia Matching Pursuit](https://en.wikipedia.org/wiki/Matching_pursuit)

## Implementation

As I said above the implementation is very close to the one used in signal processing. I will go over the steps:
    
> [!NOTE]
> For simplicity, the explanation uses a dense matrix representation. However, the actual implementation is based on sparse matrices for efficiency.

### Initializing

1. Begin with an empty `A` matrix and a target vector `b`, representing the flattened image.

2. Initialize a `candidates` matrix. Each column corresponds to the flattened representation of an image with a line drawn between two pegs. This matrix acts as the *dictionary* or `A` matrix in the [least squares approach](02_least_squares.md).

### Iterative Process

1. Extract all unused *atoms* (column vectors) from the *dictionary* (`candidates` matrix). This makes sure no line will be drawn more than once.

2. For each *atom*, append it to the `A` matrix, compute the x values, and calculate the residual.

> [!NOTE]
> After each step, only one *atom* will be selected to be appended to the `A` matrix. Once the residual is computed, the currently appended *atom* is removed before appending the next *atom*.

3. Select the *atom* that minimizes the residual and update the `A` matrix accordingly.

### Stopping Condition

The iteration continues until one of the following conditions is met:

- A predetermined number of lines have been drawn.

- The residual error stagnates (i.e., the current residual is not lower than or equal to the one at previous step).

## Heuristics

The current implementation, even when using **sparse matrices** is quite slow. To improve the time efficiency of the algorithm, I chose to integrate two heuristics.

The role of these heuristics is to reduce the set of candidate lines available for selection. If we denote the number of pegs used as `n`, at the first step, we will need to evaluate `n * (n-1) / 2` *atoms*. This results in an `A` matrix with a shape of `m * n, 1`.

For example, with a number of pegs `n = 100`, we would have `4950` possible lines/*atoms* to choose from. After selecting the first one, we will need to evaluate the remaining `4949`, and so on.

To address this problem, the following heuristics have been implemented:

### 1. Random Heuristic

This heuristic is straightforward: we randomly select the `TOP_K` candidates.

### 2. Dot Product Heuristic

Here, the dot product between each **candidate atom** and the target vector `b` is computed. The `TOP_K` candidates with the highest dot products are selected.

> [!NOTE]
> Both heuristics select an empirically tested number of candidates, `TOP_K = 10`. So, the top 10 candidates will be used for each iteration.

## Observations

- Even with the current optimizations, this approach is still quite slow. For example, using an image of shape `330 x 330` with `1000` lines to select takes approximately `5 minutes`. Compared to the **dense least squares approach**, that is **2.5 times higher**, while the **sparse least squares approach** only takes a few seconds. It's important to note that both least squares approaches select all `n * (n-1) / 2` possible lines.