# Line Rendering Comparative Analysis

## Introduction

The image below showcases each output image computed with different line rendering configurations, as well as the absolute difference from the target image and the RMS value.

![Line Rendering Results](../../outputs/experiments/line_rendering_comparative_analysis.png)

> Figure: Visual and quantitative comparison of different line rendering techniques. Each configuration varies by solver type, line drawing algorithm, number of selected lines, and coefficient representation.

## Rendering Configurations

| Figure | Solver               | Line Algorithm | Top K number of lines | Coefficient Type |
|--------|----------------------|----------------|-----------------------|------------------|
| a      | least-squares        | Bresenham      | ∞                     | float            |
| b      | least-squares        | Bresenham      | 500                   | float            |
| c      | least-squares        | Bresenham      | 500                   | binary (0/1)     |
| d      | linear-least-squares | Bresenham      | ∞                     | float            |
| e      | linear-least-squares | Bresenham      | 500                   | float            |
| f      | linear-least-squares | Bresenham      | 500                   | binary (0/1)     |
| g      | least-squares        | Xiaolin-Wu     | ∞                     | float            |
| h      | least-squares        | Xiaolin-Wu     | 500                   | float            |
| i      | least-squares        | Xiaolin-Wu     | 500                   | binary (0/1)     |
| j      | linear-least-squares | Xiaolin-Wu     | ∞                     | float            |
| k      | linear-least-squares | Xiaolin-Wu     | 500                   | float            |
| l      | linear-least-squares | Xiaolin-Wu     | 500                   | binary (0/1)     |

