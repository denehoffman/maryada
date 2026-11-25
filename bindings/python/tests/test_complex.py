import math
from typing import assert_type

import pytest

from maryada import (
    ComplexBox,
    ComplexDisk,
    ComplexDiskError,
    Interval,
    IntervalConstructionError,
)


@pytest.mark.parametrize("cls", [ComplexBox, ComplexDisk])
def test_rust_complex_aliases(cls: type[ComplexBox] | type[ComplexDisk]) -> None:
    value = cls.point(3 + 4j)
    assert value.conj() == value.conjugate()
    assert value.sqr() == value.square()
    assert value.recip() == value.reciprocal()
    assert value.powi(3) == value.pown(3)


def test_box_components_geometry_and_membership() -> None:
    box = ComplexBox(real=Interval(1, 3), imag=Interval(-1, 1))
    assert assert_type(box.real, Interval) == Interval(1, 3)
    assert box.imag == Interval(-1, 1)
    assert assert_type(box.midpoint, complex) == 2
    assert box.width == (2, 2)
    assert set(box.corners) == {1 - 1j, 1 + 1j, 3 - 1j, 3 + 1j}
    assert box.lower_corner == 1 - 1j
    assert box.upper_corner == 3 + 1j
    assert 2 + 0.5j in box
    assert 2 + 2j not in box
    assert ComplexBox(Interval(1, 2), 0) in box
    assert ComplexBox(box.real, box.imag) in box
    assert ComplexBox.EMPTY in box
    assert box.intersection(ComplexBox(Interval(0, 2), 0)) == ComplexBox(
        Interval(1, 2), 0
    )
    assert box.hull(ComplexBox.point(4)).real == Interval(1, 4)
    assert box.inflate(0, 1) == ComplexBox(Interval(0, 4), Interval(-2, 2))
    assert math.sqrt(8) <= box.diameter


def test_box_arithmetic_accepts_complex_points_and_real_intervals() -> None:
    z = ComplexBox.point(3 + 4j)
    assert assert_type(z * 1j, ComplexBox) == ComplexBox.point(-4 + 3j)
    assert 1j * z == z * 1j
    assert 1j - z == ComplexBox.point(-3 - 3j)
    assert z + Interval(-1, 1) == ComplexBox(Interval(2, 4), 4)
    assert assert_type(Interval(-1, 1) + z, ComplexBox) == z + Interval(-1, 1)
    assert z.conjugate() == ComplexBox.point(3 - 4j)
    assert assert_type(z**2, ComplexBox) == ComplexBox.point(-7 + 24j)
    assert z.square() == z**2
    assert z.mul_add(ComplexBox.I, ComplexBox.ONE) == ComplexBox.point(-3 + 3j)
    assert assert_type(abs(z), Interval) == Interval.point(5)
    modulus, phase = assert_type(z.to_polar(), tuple[Interval, Interval])
    assert modulus == abs(z)
    assert math.atan2(4, 3) in phase
    assert 2j in ComplexBox.point(-4).sqrt()


def test_disk_arithmetic_and_recentered_product() -> None:
    disk = ComplexDisk(center=1, radius=0.4375)
    assert assert_type(disk.center, complex | None) == 1
    assert assert_type(disk.radius, float | None) == 0.4375
    assert 1 + 0.25j in disk
    assert ComplexDisk(1, 0.125) in disk
    assert ComplexDisk(1, 0.4375) in disk
    assert ComplexDisk.EMPTY in disk
    assert disk + 1j == ComplexDisk(1 + 1j, 0.4375)
    assert 2 - disk == disk
    assert 1j * disk == ComplexDisk(1j, 0.4375)
    assert assert_type(
        ComplexDisk.point(3 + 4j) ** 2, ComplexDisk
    ) == ComplexDisk.point(-7 + 24j)

    ordinary = disk * disk
    recentered = disk.mul_recentered(disk)
    assert 0j in ordinary
    assert 0j not in recentered
    assert ordinary.radius is not None and recentered.radius is not None
    assert recentered.radius < ordinary.radius
    assert 2.06640625 + 0j in recentered


def test_explicit_conversions_enclose_the_original_shape() -> None:
    box = ComplexBox(Interval(-3, 3), Interval(-4, 4))
    disk = assert_type(box.to_disk(), ComplexDisk)
    assert disk.center == 0j and disk.radius == 5
    assert all(corner in disk for corner in box.corners)
    hull = assert_type(disk.to_box(), ComplexBox)
    assert hull == ComplexBox(Interval(-5, 5), Interval(-5, 5))
    assert box in hull
    assert ComplexDisk.from_interval(Interval(-1, 3)) == ComplexDisk(1, 2)
    assert ComplexBox.EMPTY.to_disk() == ComplexDisk.EMPTY
    assert ComplexDisk.ENTIRE.to_box() == ComplexBox.ENTIRE

    with pytest.raises(ComplexDiskError, match="unbounded box"):
        ComplexBox(Interval(1, math.inf), 0).to_disk()
    with pytest.raises(ComplexDiskError, match="radius"):
        ComplexDisk(0, -1)
    with pytest.raises(ComplexDiskError, match="center"):
        ComplexDisk.point(complex(math.inf, 0))
    with pytest.raises(IntervalConstructionError):
        ComplexBox.point(complex(math.nan, 0))
    with pytest.raises(TypeError):
        _ = disk + box  # ty: ignore[unsupported-operator]


def test_polar_elementary_functions_and_special_values() -> None:
    assert ComplexBox.cis(0) == ComplexBox.ONE
    assert ComplexDisk.cis(0) == ComplexDisk.ONE
    assert ComplexBox.from_polar(Interval(2, 3), 0) == ComplexBox(Interval(2, 3), 0)
    assert ComplexDisk.from_polar(Interval(2, 3), 0) == ComplexDisk(2.5, 0.5)
    assert ComplexBox.ZERO.exp() == ComplexBox.ONE
    assert ComplexDisk.ZERO.exp() == ComplexDisk.ONE
    assert ComplexDisk.EMPTY.center is None and ComplexDisk.EMPTY.radius is None
    assert ComplexDisk.ENTIRE.center is None and ComplexDisk.ENTIRE.radius == math.inf
    assert ComplexBox.EMPTY.sqrt().is_empty()
    assert ComplexDisk.EMPTY.sqrt().is_empty()
    assert ComplexDisk.ZERO.reciprocal().is_empty()


def test_repr_reconstructs_box_and_disk_values() -> None:
    namespace = {
        "ComplexBox": ComplexBox,
        "ComplexDisk": ComplexDisk,
        "Interval": Interval,
    }
    for value in (
        ComplexBox.EMPTY,
        ComplexBox.ENTIRE,
        ComplexBox(Interval(-math.inf, 3), 2),
        ComplexBox(Interval.EMPTY, 0),
        ComplexDisk.EMPTY,
        ComplexDisk.ENTIRE,
        ComplexDisk(-3 + 4j, 0.125),
    ):
        assert eval(repr(value), namespace) == value
