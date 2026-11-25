import math
from typing import assert_type

import pytest

from maryada import (
    DecoratedInterval,
    Decoration,
    Interval,
    IntervalConstructionError,
    IntervalParseError,
    IntervalParseErrorKind,
)


@pytest.mark.parametrize("cls", [Interval, DecoratedInterval])
def test_rust_real_aliases(cls: type[Interval] | type[DecoratedInterval]) -> None:
    value = cls.from_mid_rad(midpoint=2, radius=1)
    assert value == cls.from_mid_radius(2, 1)
    assert value.from_mid_rad(2, 1) == value
    assert value.sqr() == value.square()
    assert value.recip() == value.reciprocal()
    assert value.recip_split() == value.reciprocal_split()
    assert value.powi(3) == value.pown(3)


def test_checked_construction_selects_and_normalizes_decorations() -> None:
    assert DecoratedInterval(1, 2).decoration is Decoration.COM
    assert DecoratedInterval(-math.inf, 2).decoration is Decoration.DAC
    assert (
        DecoratedInterval(-math.inf, 2, decoration=Decoration.COM).decoration
        is Decoration.DAC
    )
    assert DecoratedInterval(1, 2, decoration=Decoration.ILL).is_nai()
    assert DecoratedInterval.from_mid_radius(2, 1).interval == Interval(1, 3)
    assert (
        Decoration.ILL
        < Decoration.TRV
        < Decoration.DEF
        < Decoration.DAC
        < Decoration.COM
    )
    with pytest.raises(IntervalConstructionError):
        DecoratedInterval(2, 1)
    with pytest.raises(TypeError, match="decoration must be a Decoration"):
        DecoratedInterval(1, 2, decoration=16)  # ty: ignore[invalid-argument-type]


def test_explicit_promotion_preserves_empty_and_set_membership() -> None:
    assert Interval.EMPTY.decorate() == DecoratedInterval.EMPTY
    assert (
        DecoratedInterval.from_interval(
            Interval.EMPTY, decoration=Decoration.DEF
        ).decoration
        is Decoration.TRV
    )
    bare = Interval(1, 2)
    value = assert_type(bare.decorate(Decoration.DEF), DecoratedInterval)
    assert assert_type(value.decoration, Decoration) is Decoration.DEF
    assert assert_type(value.interval, Interval) == bare
    assert value != bare.decorate()
    assert bare in value and value in bare
    assert DecoratedInterval.EMPTY in value
    assert 1 in value
    assert 1 not in DecoratedInterval.NAI
    assert Interval.EMPTY not in DecoratedInterval.NAI
    assert DecoratedInterval.NAI not in Interval.ENTIRE
    assert DecoratedInterval.NAI not in DecoratedInterval.ENTIRE
    assert DecoratedInterval.NAI.is_nai()
    assert not DecoratedInterval.NAI.is_empty()
    assert math.isnan(DecoratedInterval.NAI.lower)
    with pytest.raises(IntervalConstructionError, match="NaI has no interval part"):
        _ = DecoratedInterval.NAI.interval


def test_arithmetic_propagates_decorations_and_promotes_bare_operands() -> None:
    left = DecoratedInterval(1, 2, decoration=Decoration.DAC)
    right = DecoratedInterval(3, 4, decoration=Decoration.DEF)
    assert (left + right).decoration is Decoration.DEF
    assert left.mul_add(right, DecoratedInterval.ONE).decoration is Decoration.DEF
    assert DecoratedInterval(-1, 4).sqrt().decoration is Decoration.TRV
    assert DecoratedInterval(-1, 1).sign().decoration is Decoration.DEF
    assert left.intersection(right).is_empty()
    assert left.intersection(left).decoration is Decoration.TRV

    bare = Interval(3, 4)
    forward = assert_type(left + bare, DecoratedInterval)
    reflected = assert_type(bare + left, DecoratedInterval)
    assert forward == reflected == DecoratedInterval(4, 6, decoration=Decoration.DAC)
    assert bare - left == DecoratedInterval(1, 3, decoration=Decoration.DAC)
    assert left - bare == DecoratedInterval(-3, -1, decoration=Decoration.DAC)
    assert (bare + DecoratedInterval.NAI).is_nai()
    assert (DecoratedInterval.NAI + bare).is_nai()
    assert (3 * DecoratedInterval.NAI).is_nai()


def test_split_and_reductions_preserve_decorated_results() -> None:
    pieces = assert_type(
        DecoratedInterval(-1, 1).reciprocal_split(), tuple[DecoratedInterval, ...]
    )
    assert tuple(piece.interval for piece in pieces) == (
        Interval(-math.inf, -1),
        Interval(1, math.inf),
    )
    assert all(piece.decoration is Decoration.TRV for piece in pieces)
    assert DecoratedInterval.NAI.reciprocal_split() == (DecoratedInterval.NAI,)
    terms = [
        Interval.point(2**53),
        DecoratedInterval(1, 1, decoration=Decoration.DEF),
        Interval.point(-(2**53)),
    ]
    exact = DecoratedInterval(1, 1, decoration=Decoration.DEF)
    assert DecoratedInterval.sum(iter(terms)) == exact
    assert DecoratedInterval.dot(terms, (Interval.ONE for _ in terms)) == exact


@pytest.mark.parametrize(
    "value",
    [
        DecoratedInterval.EMPTY,
        DecoratedInterval.NAI,
        DecoratedInterval.ENTIRE,
        DecoratedInterval(1, 2, decoration=Decoration.DEF),
        DecoratedInterval(-math.inf, 2, decoration=Decoration.TRV),
        DecoratedInterval.point(math.nextafter(0.0, math.inf)),
    ],
)
def test_decorated_text_binary_and_repr_round_trips(value: DecoratedInterval) -> None:
    assert DecoratedInterval.parse(value.to_text()) == value
    assert DecoratedInterval.from_bytes(value.to_bytes()) == value
    assert DecoratedInterval.from_bytes(value.to_bytes("little"), "little") == value
    assert len(value.to_bytes()) == 17
    assert value.to_bytes()[-1] == value.decoration.value
    restored = eval(
        repr(value), {"DecoratedInterval": DecoratedInterval, "Decoration": Decoration}
    )
    assert restored == value


def test_invalid_binary_and_distinct_special_text_are_typed() -> None:
    with pytest.raises(IntervalConstructionError):
        DecoratedInterval.from_bytes(Interval.EMPTY.to_bytes())
    assert DecoratedInterval.parse("[empty]_trv") == DecoratedInterval.EMPTY
    assert DecoratedInterval.parse("[nai]") == DecoratedInterval.NAI
    assert DecoratedInterval.NAI.to_decimal(3) == "[nai]"
    result, possibly_undefined = assert_type(
        DecoratedInterval.parse_checked("[0.100000000000000001,1/10]_def"),
        tuple[DecoratedInterval, bool],
    )
    assert possibly_undefined and result.decoration is Decoration.DEF


def test_decorated_parse_exposes_rust_diagnostics() -> None:
    with pytest.raises(IntervalParseError) as raised:
        DecoratedInterval.parse_checked("[1]_invalid")
    assert raised.value.kind is IntervalParseErrorKind.INVALID_DECORATION
    assert raised.value.byte_offset == 4
    with pytest.raises(IntervalParseError) as ambiguous:
        DecoratedInterval.parse("[0.100000000000000001,1/10]_def")
    assert ambiguous.value.kind is IntervalParseErrorKind.POSSIBLY_UNDEFINED
    assert ambiguous.value.byte_offset is None
