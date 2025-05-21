# Least Squares Regularized

Considering our previous [Quadratic Problem Formulation](./11_binary_projection_ls.md#quadratic-programming-formulation), I can add regularization terms to the CVXOPT solver in order to push values toward 0 or 1. This is useful when I want to encourage binary-like solutions without explicitly constraining the problem to binary values.

The base optimization problem is:

```math
\min_{x} \frac{1}{2} x^TPx + q^Tx \text{ subject to } x \in [0,1]^n
```

I extend this problem with two types of regularizers: **smooth** and **absolute**.

## Smooth Regularization

### Formula

```math
\lambda 4x(1-x)
```

<img src="../../docs/assets/smooth_regularizer_graph.png" width=400 alt="Smooth Regularizer Graph">

> Smooth Regularizer Graph

This regularizer promotes binary values by penalizing intermediate values in `[0,1]`. The function 4x(1−x) is zero at x=0 and x=1, and reaches its maximum at x=0.5, making it ideal for encouraging binary-like outputs.

### Modified Objective

The regularized objective becomes:

```math
\min_{x} \frac{1}{2} x^T(P - 8\lambdaI)x + (q + 4\lambda1)^Tx \text{ subject to } x \in [0,1]^n
```

This modification adjusts both the quadratic and linear terms. The scalar λ controls the strength of the regularization. A safeguard is included to ensure the matrix remains positive semi-definite by limiting:

```math
\lambda \leq \frac{\lambda_{min}(P)}{8}
```

```math
\text{where}
```

```math
P = 2A^TA
```

```math
q = -2A^Tb
```

## ABS Regularization

### Formula

```math
\lambda 1 - 2\abs{x - 0.5}
```

This regularizer penalizes values close to 0.5 and rewards values closer to 0 or 1, forming a piecewise linear "V" shape centered at 0.5.

<img src="../../docs/assets/abs_regularizer_graph.png" width=400 alt="ABS Regularizer Graph">

> ABS Regularizer Graph

### Reformulation

To express the absolute value in a quadratic program, I introduce auxiliary variables:

```math
t_i \geq \abs{x_i - 0.5}
```

And optimize over:

```math
z = \left[ x; t \right] \in \mathbb{R}^{2n}
```

### Modified Objective

```math
\min_{x, t} \frac{1}{2} x^TPx + q^Tx + 2\lambda \sum_i{t_i} \text{ subject to } x \in [0,1]^n
```

### Constraints

The constraints ensure valid bounds on x and encode:

```math
t_i \geq \abs{x_i - 0.5}
```

```math
0 \leq x \geq 1
```

```math
t \geq x - 0.5
```

```math
t \geq -x + 0.5
```

These are incorporated into the QP using inequality constraints on the concatenated variable `z = [x;t]`

[//]: # (TODO: add binary regularizer)

### X Coefficient Histogram

![X Coefficient Histogram](../../outputs/experiments/regularized_x_binned.png)

[//]: # (TODO: conclusion)

## Results

![No Regularizer](../../benchmarks/img_outputs/least_squares_regularized/image_000.png)
![Smooth Regularizer](../../benchmarks/img_outputs/least_squares_regularized/image_001.png)
![ABS Regularizer](../../benchmarks/img_outputs/least_squares_regularized/image_002.png)

Comparison of solutions: no regularization (left), smooth regularization (center), and absolute regularization (right).

![Difference Images](../plots/least_squares_regularized/Difference%20Images.png)

> Difference Images

![Time Usage](../plots/least_squares_regularized/Time%20Usage.png)

> Time Usage

![Memory Usage](../plots/least_squares_regularized/Memory%20Usage.png)

> Memory Usage

