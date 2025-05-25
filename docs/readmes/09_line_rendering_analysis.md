# Line Rendering Comparative Analysis

## Introduction

The image below showcases each output image computed with different line rendering configurations, as well as the absolute difference from the target image and the RMS value.

![Line Rendering Results](../../outputs/experiments/line_rendering_comparative_analysis.png)

> Figure: Visual and quantitative comparison of different line rendering techniques. Each configuration varies by solver type, line drawing algorithm, number of selected lines, and coefficient representation.

## Rendering Configurations

| Figure | Solver | Line Algorithm | Top K number of lines | Coefficient Type |
|--------|--------|----------------|-----------------------|------------------|
| a      | ls     | Bresenham      | ∞                     | float            |
| b      | ls     | Bresenham      | 500                   | float            |
| c      | ls     | Bresenham      | 500                   | binary (0/1)     |
| d      | lls    | Bresenham      | ∞                     | float            |
| e      | lls    | Bresenham      | 500                   | float            |
| f      | lls    | Bresenham      | 500                   | binary (0/1)     |
| g      | ls     | Xiaolin-Wu     | ∞                     | float            |
| h      | ls     | Xiaolin-Wu     | 500                   | float            |
| i      | ls     | Xiaolin-Wu     | 500                   | binary (0/1)     |
| j      | lls    | Xiaolin-Wu     | ∞                     | float            |
| k      | lls    | Xiaolin-Wu     | 500                   | float            |
| l      | lls    | Xiaolin-Wu     | 500                   | binary (0/1)     |

Abbreviations:
- **ls** – Least Squares
- **lls** – Linear Least Squares