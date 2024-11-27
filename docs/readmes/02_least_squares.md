# Least Squares

## Terminology

The **Least Squares** approach minimizes the aforementioned function:

```math
\min{\| M \cdot X - Y \|}_2^2
```

by the vector x such that the l<sup>2</sup>-norm between the ground truth image (initial image) and
the computed image (the one calculated using this algorithm) is minimum.

Each column in the matrix `M` represents the flattened row-wise matrix of an image where a line is drawn connecting two pegs.
Currently, `M` is stored as a **dense** matrix, but a **sparse** representation will also be explored and benchmarked for comparison.

The vector `X` contains values between 0 and 1, indicating the intensity or extent to which each line is drawn.

The vector `Y` is the flattened row-wise representation of the initial image.

## Peg Placement

As for the placing of the pegs in a circular shape at equidistant space between them, I used the parametric
equation for a circle. 

```math
\text{For a circle centered in } (x_c, y_c) \text{ with radius } r \text{ and } n \text{ pegs}:
```
```math
x_k = x_c + r \cdot \cos(\frac{2 \pi k}{n})
```
```math
y_k = y_c + r \cdot \sin(\frac{2 \pi k}{n})
```
```math
\text{for } k = 0, 1, 2, ..., n-1
```

## Bresenham

To draw each line in the matrix, the Bresenham algorithm was used, determining the points needed to be
selected between two chosen pegs in order to have a close represantation of a straight drawn line.

## Output Image

After solving the system for `X`, we can generate the output image using:

```math
O = MX
```

## Observations

However, since some values fall outside the `[0, 1]` range, I applied clipping to validate the constraint.
Additionally, these values are **real** numbers, meaning `X` represents an intensity vector rather than a binary one.

We could also experiment with allowing negative values, which would correspond to subtracting a line, 
or **scaling** the values to fit within `[0, 1]` instead of **clipping**. 

However, keep in mind that subtracting an edge violates physical constraints and is only possible digitally.