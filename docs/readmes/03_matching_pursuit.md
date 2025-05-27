# Matching Pursuit (MP)

## Terminology

**"Matching Pursuit"** is a specific greedy algorithm applied in signal processing and machine learning to approximate a signal using a linear combination of *dictionary atoms* (columns from the matrix)

It iteratively selects the *atom* that at the current step reduces the residual most significantly.

This concept has been adopted and integrated into this context to explore its performance and behaviour.

You can read more here: [Wikipedia Matching Pursuit](https://en.wikipedia.org/wiki/Matching_pursuit)

## Implementation Greedy Method

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

### Heuristics

The current implementation, even when using **sparse matrices** is quite slow. To improve the time efficiency of the algorithm, I chose to integrate two heuristics.

The role of these heuristics is to reduce the set of candidate lines available for selection. If we denote the number of pegs used as `N`, at the first step, we will need to evaluate `N * (N-1) / 2` *atoms*.
This results in an `A` matrix with a shape of `m * m, 1`; at the second step, `m * m, 2`; and so on, until the final step, where the shape will be `m * m, N * (N-1) / 2`.

For example, with a number of pegs `N = 100`, we would have `4950` possible lines/*atoms* to choose from. After selecting the first one, we will need to evaluate the remaining `4949`, and so on.

To address this problem, the following heuristics have been implemented:

#### 1. Random Heuristic

This heuristic is straightforward: we randomly select the `TOP_K` candidates.

#### 2. Dot Product Heuristic

Here, the dot product between each **candidate atom** and the target vector `b` is computed. The `TOP_K` candidates with the highest dot products are selected.

> [!NOTE]
> Both heuristics select an empirically tested number of candidates, `TOP_K = 10`. So, the top 10 candidates will be used for each iteration.

### Observations

- Even with the current optimizations, this approach is still quite slow. For example, using an image of shape `330 x 330` with `1000` lines to select takes approximately `5 minutes`. Compared to the **dense least squares approach**, that is **2.5 times higher**, while the **sparse least squares approach** only takes a few seconds. It's important to note that both least squares approaches select all `N * (N-1) / 2` possible lines.

## Implementation Orthogonal Matching Pursuit (OMP)

The initialization phase of the Orthogonal Matching Pursuit (OMP) method is similar to the greedy approach, but the iterative steps involve a key difference that makes **OMP** more efficient.

In the **OMP** method, the algorithm proceeds as follows:

1. **Selection of Atoms:** At each iteration, we select the **atom** from the **dictionary** that has the greatest correlation with the current residual vector `b`. This is done by calculating the dot product between `b` and each **atom** in the **dictionary**. The **atom** with the largest dot product is chosen, similar to our **Dot Product Heuristic**.
2. **Update the Residual:** Recompute the **Least Squares** solution after each selection. **OMP** updates the residual vector `b` by subtracting the contribution of the selected **atom** from it. This step reduces the residual error in the approximation.
3. **Repeat the Process:** The procedure repeats for a set number of iterations or until the residual no longer decreases significantly. At each iteration, the selected **atom** is added to the active set, and the residual is updated.

The key advantage of **OMP** over the original greedy methods is that it avoids repeatedly solving the **least squares** problem for each candidate **atom**. Instead, it only updates the residual, which significantly speeds up the computation.