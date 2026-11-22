import math
import operator
import struct
import sys
from collections.abc import Callable
from fractions import Fraction
from typing import assert_type

import pytest

from maryada import Interval, IntervalConstructionError, IntervalParseError


def test_construction_and_typing() -> None:
    x = assert_type(Interval(lower=1, upper=2.0), Interval)
    assert assert_type(x.bounds, tuple[float, float]) == (1.0, 2.0)
    assert assert_type(x.lower, float) == 1.0
    assert assert_type(x.upper, float) == 2.0
    assert assert_type(1 + x, Interval) == Interval(2, 3)
    assert assert_type(x / 2, Interval) == Interval(0.5, 1)
    assert assert_type(x.sqrt(), Interval).lower == 1.0
    assert Interval.from_mid_radius(1.5, 0.5) == x


@pytest.mark.parametrize(
    ("lower", "upper"),
    [(2.0, 1.0), (math.nan, 1.0), (math.inf, math.inf), (-math.inf, -math.inf)],
)
def test_invalid_numeric_construction_raises_typed_error(
    lower: float, upper: float
) -> None:
    with pytest.raises(IntervalConstructionError):
        Interval(lower, upper)
    with pytest.raises(IntervalConstructionError):
        Interval.from_mid_radius(0, -1)
    with pytest.raises(IntervalConstructionError):
        Interval.point(math.inf)


def test_integer_coercion_checks_exactness() -> None:
    exact = 2**100
    assert Interval.point(exact).bounds == (float(exact), float(exact))
    with pytest.raises(OverflowError):
        Interval.point(2**53 + 1)
    with pytest.raises(OverflowError):
        Interval.point(2**2000)
    with pytest.raises(OverflowError):
        _ = Interval.point(0) + (2**53 + 1)


def test_membership_compares_integers_without_rounding() -> None:
    assert 0 in Interval(-1, 1)
    assert 2**53 + 1 not in Interval.point(2**53)
    assert 2**2000 in Interval.entire()
    assert -(2**2000) in Interval(-math.inf, 0)
    assert math.inf not in Interval.entire()
    assert not operator.contains(Interval.entire(), math.nan)
    assert "1" not in Interval.entire()
    assert 0 not in Interval.empty()


def test_interval_membership_is_inclusive_set_containment() -> None:
    outer = Interval(0, 3)
    assert Interval(0, 2) in outer
    assert Interval(0, 3) in outer
    assert Interval(2, 4) not in outer
    assert outer not in Interval(0, 2)
    assert Interval.EMPTY in outer
    assert Interval.EMPTY in Interval.EMPTY
    assert outer not in Interval.EMPTY
    assert outer in Interval.ENTIRE


def test_intervals_are_immutable_and_unsupported_operands_use_dispatch() -> None:
    class Reflected:
        def __radd__(self, left: Interval) -> Interval:
            return left + 2

    x = Interval(1, 2)
    with pytest.raises(AttributeError):
        x.lower = 0  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        operator.add(x, "1")  # ty: ignore[no-matching-overload]
    assert operator.add(x, Reflected()) == Interval(3, 4)


def test_mixed_arithmetic_and_sets() -> None:
    x = Interval(2, 4)
    y = Interval(4, 8)
    assert x + y == Interval(6, 12)
    assert x + 2 == 2 + x == Interval(4, 6)
    assert x - 2 == Interval(0, 2)
    assert 2 - x == Interval(-2, 0)
    assert x * 2 == 2 * x == Interval(4, 8)
    assert x / 2 == Interval(1, 2)
    assert 8 / x == Interval(2, 4)
    assert -x == Interval(-4, -2)
    assert abs(Interval(-4, 2)) == Interval(0, 4)
    assert x.intersection(y) == Interval.point(4)
    assert x.hull(y) == Interval(2, 8)
    assert x.subset(Interval.entire())
    assert (x / 0).is_empty()
    assert Interval(-1, 1).reciprocal().is_entire()
    assert x + Interval.empty() == Interval.empty()


def test_decimal_parsing_and_arithmetic_enclose_exact_values() -> None:
    point = Interval.point(0.1)
    assert Fraction(point.lower) > Fraction("0.1")
    x = Interval.parse("[0.1]")
    assert Fraction(x.lower) <= Fraction("0.1") <= Fraction(x.upper)
    assert Interval.parse("[0.1, 0.2]").upper == 0.2
    result = Interval.point(0.1) + Interval.point(0.2)
    exact = Fraction(0.1) + Fraction(0.2)
    assert Fraction(result.lower) <= exact <= Fraction(result.upper)


def test_parse_errors_are_distinct_from_empty() -> None:
    assert Interval.parse("[empty]").is_empty()
    with pytest.raises(IntervalParseError, match="at byte 3"):
        Interval.parse("[2,1]")
    with pytest.raises(IntervalParseError, match="exact order"):
        Interval.parse("[0.100000000000000001,1/10]")


def test_repr_uses_python_constructors() -> None:
    assert repr(Interval(0, 1)) == "Interval(0.0, 1.0)"
    assert repr(Interval.empty()) == "Interval.empty()"
    assert repr(Interval.entire()) == "Interval.entire()"
    assert repr(Interval(-math.inf, 3)) == "Interval(float('-inf'), 3.0)"
    assert repr(Interval(3, math.inf)) == "Interval(3.0, float('inf'))"


@pytest.mark.parametrize(
    "value",
    [
        Interval.empty(),
        Interval.entire(),
        Interval.point(0),
        Interval(-math.inf, 3),
        Interval(-sys.float_info.max, sys.float_info.max),
        Interval.point(math.nextafter(0, math.inf)),
    ],
)
def test_exact_text_and_repr_round_trip_endpoint_bits(value: Interval) -> None:
    expected = struct.pack("dd", *value.bounds)
    assert struct.pack("dd", *Interval.parse(value.to_text()).bounds) == expected
    reconstructed = eval(repr(value), {"Interval": Interval})
    assert isinstance(reconstructed, Interval)
    assert struct.pack("dd", *reconstructed.bounds) == expected


@pytest.mark.parametrize(
    ("operation", "reference"),
    [
        (Interval.exp, 1.0),
        (Interval.sin, 0.0),
        (Interval.cos, 1.0),
        (Interval.tan, 0.0),
    ],
)
def test_elementary_functions(
    operation: Callable[[Interval], Interval], reference: float
) -> None:
    assert reference in operation(Interval.point(0))
    assert Interval(-2, -1).sqrt().is_empty()
    assert Interval.point(1).log() == Interval.point(0)


def test_alternate_constructors_follow_python_staticmethod_lookup() -> None:
    value = Interval(0, 1)
    assert assert_type(value.point(3), Interval) == Interval.point(3)
    assert value.parse("[0,1]") == value
    assert value.from_mid_radius(0.5, 0.5) == value
    assert value.empty() == Interval.EMPTY
    assert value.entire() == Interval.ENTIRE


def test_geometry_and_comparisons_use_set_inclusion() -> None:
    x = Interval(-1, 3)
    y = Interval(0, 1)
    assert assert_type(x.midpoint, float) == 1
    assert x.width == 4
    assert x.radius == x.inner_radius == 2
    assert assert_type(x.midpoint_radius, tuple[float, float]) == (1, 2)
    assert x.magnitude == 3 and x.mignitude == 0
    assert x.is_bounded() and not x.is_nai() and not x.is_singleton()
    assert Interval.ONE.is_singleton() and math.pi in Interval.PI
    assert y < x and y <= x and x > y and x >= y
    assert y.interior(x) and x.intersects(y) and not x.disjoint(y)
    overlap = Interval(2, 4)
    assert x.intersects(overlap) and not x <= overlap and not overlap <= x
    assert x.disjoint(Interval(4, 5))
    assert x.bisect() == (Interval(-1, 1), Interval(1, 3))
    assert x.hull_value(4) == Interval(-1, 4)
    assert x.inflate(0, 1) == Interval(-2, 4)


def test_powers_rounding_and_pairwise_operations() -> None:
    x = Interval(-2, 1)
    assert assert_type(x**2, Interval) == x.square() == x.pown(2) == Interval(0, 4)
    assert Interval(2, 3).subset(Interval(4, 9) ** 0.5)
    assert 8 in assert_type(2 ** Interval.point(3), Interval)
    assert Interval(1, 4).subset(Interval(1, 2).pow(Interval.point(2)))
    assert x.sign() == Interval(-1, 1)
    halves = Interval(-0.5, 0.5)
    assert halves.round_ties_to_even() == Interval.ZERO
    assert halves.round_ties_to_away() == Interval(-1, 1)
    assert halves.ceil() == Interval(0, 1)
    assert halves.floor() == Interval(-1, 0)
    assert halves.trunc() == Interval.ZERO
    y = Interval(1, 4)
    assert y.min(Interval(2, 3)) == Interval(1, 3)
    assert y.max(Interval(2, 3)) == Interval(2, 4)
    assert y.cancel_minus(Interval(1, 2)) == Interval(0, 2)
    assert y.cancel_plus(Interval(1, 2)) == Interval(3, 5)
    assert Interval.point(3).hypot(Interval.point(4)) == Interval.point(5)
    assert Interval.ZERO.atan2(Interval.ONE) == Interval.ZERO
    assert y.mul_add(Interval.point(2), Interval.ONE) == Interval(3, 9)


def test_disconnected_results_and_exact_reductions() -> None:
    x = Interval(-1, 1)
    assert assert_type(x.reciprocal_split(), tuple[Interval, ...]) == (
        Interval(-math.inf, -1),
        Interval(1, math.inf),
    )
    assert Interval(2, 4).div_split(x) == (
        Interval(-math.inf, -2),
        Interval(2, math.inf),
    )
    assert Interval.ZERO.reciprocal_split() == ()
    terms = [Interval.point(2**53), Interval.ONE, Interval.point(-(2**53))]
    assert Interval.sum(iter(terms)) == Interval.ONE
    assert Interval.dot(terms, (Interval.ONE for _ in terms)) == Interval.ONE
    with pytest.raises(ValueError):
        Interval.dot(terms, [])


def test_serialization_and_checked_parse() -> None:
    x = Interval.parse("[1.26, 1.34]")
    assert x.to_decimal(1) == "[1.2,1.4]"
    assert x.subset(Interval.parse(x.to_decimal(2, notation="scientific")))
    assert x.subset(Interval.parse(x.to_decimal(1, notation="center_radius")))
    assert Interval.from_bytes(x.to_bytes()) == x
    assert Interval.from_bytes(x.to_bytes("little"), byteorder="little") == x
    assert Interval.from_bytes(Interval.empty().to_bytes()).is_empty()
    with pytest.raises(IntervalConstructionError):
        Interval.from_bytes(b"invalid")
    with pytest.raises(ValueError):
        x.to_decimal(0, notation="scientific")
    parsed, possibly_undefined = assert_type(
        Interval.parse_checked("[0.100000000000000001,1/10]"), tuple[Interval, bool]
    )
    assert possibly_undefined and 0.1 in parsed


def test_extended_elementary_functions() -> None:
    for operation in (Interval.exp2, Interval.exp10):
        assert operation(Interval.ZERO) == Interval.ONE
    for operation in (Interval.log2, Interval.log10, Interval.acosh, Interval.acos):
        assert 0 in operation(Interval.ONE)
    for operation in (
        Interval.expm1,
        Interval.log1p,
        Interval.asin,
        Interval.atan,
        Interval.sinh,
        Interval.tanh,
        Interval.asinh,
        Interval.atanh,
    ):
        assert 0 in operation(Interval.ZERO)
    assert 1 in Interval.ZERO.cosh()
