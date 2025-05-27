# Binary Projection Least Squares Regularized (BPLSR)

In the [**Least Squares Regularized**](./12_least_squares_regularized.md) readme,it's noted that introducing a binary-enforcing regularization term can break the convexity of the optimization problem. To address this, a new approach was proposed: instead of adding the regularization term explicitly to the objective, I incorporate its effect directly into the cost matrix **before** solving the system. This preserves convexity while still encouraging binary-like solutions through adaptive weighting.

The base optimization problem is:

```math
\min_{x} \frac{1}{2} x^TPx + q^Tx \text{ subject to } x \in [0,1]^n
```

## Methodology

I introduce the following vector: `w`

```math
w_i = x_{i-1}(1-x_{i-1}) + \epsilon
```

This vector forms the diagonal of a regularization matrix `W = diag(w)`, which is used to modify the original quadratic cost matrix `P`. The updated regularized problem becomes:

```math
\min_{x} \frac{1}{2} x^T(P + \lambda W)x + q^Tx \text{ subject to } x \in [0,1]^n
```

where `λ` controls the strength of the regularization.

### Key Steps

#### 1. Initialization
- Start with a uniform weight matrix `W = I`.

#### 2. Weight Update
- At each iteration (or after solving once), update the weights using the formula:

```math
w_i = x_{i-1}(1-x_{i-1}) + \epsilon
```

#### 3. Solve System
- Solve the system 

```math
\min_{x} \frac{1}{2} x^T(P + \lambda W)x + q^Tx \text{ subject to } x \in [0,1]^n
```

- After solving, project variables into a binary candidate set `set1` as described in [**Binary Projection Least Squares**](./11_binary_projection_ls.md)

#### Iterative Refinement

- Repeat steps 2–3 iteratively until convergence criteria are met (e.g., the solution error or objective value stops decreasing).

