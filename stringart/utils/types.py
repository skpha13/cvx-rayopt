from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Point = namedtuple("Point", ["x", "y"])

CropMode = Literal["first-half", "center", "second-half"]

MatrixRepresentation = Literal["dense", "sparse"]

MatchingPursuitMethod = Literal["greedy", "orthogonal"]

Rasterization = Literal["bresenham", "xiaolin-wu"]

SolverType = Literal["least-squares", "linear-least-squares", "matching-pursuit"]


@dataclass
class Metadata:
    path: Path
