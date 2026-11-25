"""Structured diagnostics for interval literals."""

from enum import StrEnum


class IntervalParseErrorKind(StrEnum):
    """The reason an interval literal was rejected."""

    INVALID_SYNTAX = "invalid_syntax"
    INVALID_NUMBER = "invalid_number"
    INVALID_DECORATION = "invalid_decoration"
    REVERSED_BOUNDS = "reversed_bounds"
    INVALID_INFINITY = "invalid_infinity"
    INVALID_UNCERTAINTY = "invalid_uncertainty"
    POSSIBLY_UNDEFINED = "possibly_undefined"


class IntervalParseError(ValueError):
    """An invalid or ambiguous interval literal.

    byte_offset is a zero-based UTF-8 offset into the original input.
    It is None when the parser cannot certify the order of the bounds.
    """

    kind: IntervalParseErrorKind
    byte_offset: int | None

    def __init__(
        self,
        message: str = "",
        *,
        kind: IntervalParseErrorKind = IntervalParseErrorKind.INVALID_SYNTAX,
        byte_offset: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.byte_offset = byte_offset
