"""IEEE 1788 interval decorations."""

from enum import IntEnum


class Decoration(IntEnum):
    """Definedness and continuity, ordered from weakest to strongest."""

    ILL = 0
    TRV = 4
    DEF = 8
    DAC = 12
    COM = 16
