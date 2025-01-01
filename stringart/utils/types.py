from collections import namedtuple
from typing import Literal

Point = namedtuple("Point", ["x", "y"])

Mode = Literal["first-half", "center", "second-half"]
