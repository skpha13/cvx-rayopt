from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

Point = namedtuple("Point", ["x", "y"])

CropMode = Literal["first-half", "center", "second-half"]

MatrixRepresentation = Literal["dense", "sparse"]

MatchingPursuitMethod = Literal["greedy", "orthogonal"]

Rasterization = Literal["bresenham", "xiaolin-wu"]

SolverType = Literal[
    "least-squares", "linear-least-squares", "matching-pursuit", "binary-projection-ls", "least-squares-regularized"
]

QPSolvers = Literal["scipy", "cvxopt"]

RegularizationType = Literal["smooth", "abs"]


@dataclass
class Metadata:
    path: Path


class ElapsedTime(TypedDict):
    hours: int
    minutes: int
    seconds: int
    milliseconds: int


class MemorySize(TypedDict):
    gigabytes: int
    megabytes: int
    kilobytes: int
    bytes: int
