"""Outward-rounded binary64 interval arithmetic."""

from ._decoration import Decoration as Decoration
from ._native import DecoratedInterval as DecoratedInterval
from ._native import (
    Interval as Interval,
)
from ._native import (
    IntervalConstructionError as IntervalConstructionError,
)
from ._native import (
    IntervalParseError as IntervalParseError,
)

__all__ = [
    "DecoratedInterval",
    "Decoration",
    "Interval",
    "IntervalConstructionError",
    "IntervalParseError",
]
