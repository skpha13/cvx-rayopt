from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Point = namedtuple("Point", ["x", "y"])

Mode = Literal["first-half", "center", "second-half"]

Method = Literal["dense", "sparse"]

MatchingPursuitMethod = Literal["greedy", "orthogonal"]


@dataclass
class Metadata:
    path: Path
