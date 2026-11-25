# maryada for Python

maryada provides outward-rounded binary64 intervals for CPython 3.11 and later.
An `Interval` is immutable. Arithmetic returns a new interval enclosing the
results of applying the operation to all defined input values.

```python
from maryada import Interval

x = Interval.parse("[0.1, 0.2]")
y = 2 * x + 1

assert 1.25 in y
assert Interval.parse(y.to_text()) == y
print(y.bounds)
```

`Interval(lower, upper)` takes the binary64 values Python passes.
`Interval.parse("[0.1, 0.2]")` rounds the exact decimal endpoints outward.
`Interval.point(value)` constructs a finite singleton, and
`Interval.from_mid_radius(midpoint, radius)` rounds the computed endpoints
outward. Construction and scalar arithmetic accept floats and integers that
fit exactly in binary64. An integer that would round raises `OverflowError`.
Membership tests compare integers exactly, including integers larger than
binary64 can represent.

Invalid bounds raise `IntervalConstructionError`. Invalid interval text or
text with ambiguous endpoint order raises `IntervalParseError`. Both exceptions
are subclasses of `ValueError`. A valid empty set is represented by
`Interval.empty()`. For example, `Interval(-2, -1).sqrt()` is empty because
the input has no nonnegative members. `Interval.entire()` represents all real
numbers.

`IntervalParseError.kind` is an `IntervalParseErrorKind` enum member.
`byte_offset` is a zero-based UTF-8 byte offset into the original text.
For a rejected ambiguous ordering, the kind is `POSSIBLY_UNDEFINED` and
the offset is `None`. Both real interval types expose these diagnostics.

Call alternate constructors on the class, for example `Interval.point(3)`,
`Interval.parse("...")`, and `Interval.from_bytes(data)`. They use Python's
static method lookup, so CPython's REPL can also suggest them on instances.

## Scalar operations

- Use properties such as `bounds`, `midpoint`, `width`, `radius`, and
  `magnitude` to inspect an interval. The constants `Interval.ZERO`,
  `Interval.ONE`, `Interval.PI`, `Interval.TAU`, `Interval.E`, and
  `Interval.LN_2` are ready to use.
- Arithmetic includes `**`, `square()`, and fused `mul_add()`, together with
  the Rust scalar elementary functions and pointwise rounding operations.
  Integer `**` uses the signed 32-bit integer-power operation. Float and
  interval exponents use the general power enclosure.
- Comparisons use set inclusion. For example, `x <= y` means every point in
  `x` belongs to `y`. Use `intersection()`, `hull()`, or `bisect()` to
  construct related enclosures.
- Dividing through zero can give two disjoint pieces. Use `div_split()` or
  `reciprocal_split()` to get a tuple containing the individual components.
  `Interval.sum(terms)` and `Interval.dot(left, right)` use the Rust exact
  accumulation routines.
- `to_bytes()` and `Interval.from_bytes()` round-trip the exact IEEE
  binary encoding. `to_decimal(digits, notation=...)` formats an
  outward-rounded decimal enclosure. `Interval.parse_checked(text)` returns
  the parsed interval and a boolean indicating ambiguous endpoint order.

Use `to_text()` for an exact hexadecimal literal, `repr(interval)` for a Python
constructor expression, and `str(interval)` for display.
Bare interval equality compares sets. Use `in` for point membership or interval
containment. For intervals, `a in b` is equivalent to `a.subset(b)`, including
when both intervals are equal or `a` is empty.

The [weighted-sum example](examples/weighted_sum.py) uses `Interval.dot()`
to check a budget against uncertain usage and prints the exact interval
literal for storage. Run it from the checkout after installing the package.

```sh
python bindings/python/examples/weighted_sum.py
```

## Decorated intervals

`DecoratedInterval` provides the same scalar operations and carries an IEEE 1788
decoration that records definedness and continuity. The `Decoration` enum runs
from weakest to strongest as `ILL`, `TRV`, `DEF`, `DAC`, and `COM`.

```python
from maryada import DecoratedInterval, Decoration, Interval

x = DecoratedInterval(-1, 4)
assert x.sqrt().decoration is Decoration.TRV

y = Interval(1, 2).decorate(Decoration.DEF)
assert (1 + y).decoration is Decoration.DEF
assert y.interval == Interval(1, 2)
```

In mixed arithmetic, a bare operand gets its strongest valid decoration and
the result is decorated. Equality between decorated values compares both
bounds and decoration. Membership compares the represented sets.

`DecoratedInterval.empty()` represents the empty set, and
`DecoratedInterval.nai()` represents Not an Interval. NaI propagates through
arithmetic and raises `IntervalConstructionError` when you access `.interval`.
Exact text and binary round-trips preserve the decoration and both special
values.

## Complex enclosures

`ComplexBox` encloses independent real and imaginary intervals.
`ComplexDisk` encloses values within a radius of a complex center. Both
support complex arithmetic, elementary functions, modulus and argument
bounds, and set containment. Use `abs(z)` for a real interval enclosing the
modulus.

Both types accept finite binary64 complex points in arithmetic and membership
tests. Box arithmetic also accepts a bare `Interval` along the real axis.
`a in b` checks inclusive containment for two boxes or two disks.

```python
from maryada import ComplexBox, ComplexDisk, Interval

box = ComplexBox(Interval(1, 2), Interval(-1, 1))
assert 1.5 + 0.5j in box
assert box in box.to_disk().to_box()

disk = ComplexDisk(1 + 0j, 0.4375)
assert 0j in disk * disk
assert 0j not in disk.mul_recentered(disk)
```

Use `to_box()` for a disk's rectangular hull and `to_disk()` for a box's
enclosing disk. Both conversions can widen the enclosure. A proper
unbounded box or an overflowing radius raises `ComplexDiskError`.
Ordinary disk multiplication uses the fixed-center product.
`mul_recentered()` spends extra work to try a smaller radius.

## Rust method names

`sqr()`, `recip()`, and `powi()` are aliases for `square()`, `reciprocal()`,
and `pown()`. Complex enclosures also support `conj()` for `conjugate()`.
Real intervals support `recip_split()` and `from_mid_rad()` alongside
`reciprocal_split()` and `from_mid_radius()`.

Rust inspection methods such as `inf()`, `mid()`, and `wid()` correspond
to Python properties such as `lower`, `midpoint`, and `width`.

## Develop

From the repository root, use uv to install the package and test dependencies
into a virtual environment.

```sh
uv venv
uv pip install -e . --group test
uv run --no-project pytest
uv run --no-project ty check
uv run --no-project ruff check bindings/python
uv run --no-project ruff format --check bindings/python
```

The package ships typing information through `py.typed` and the
`_native.pyi` stub. Update the stub alongside changes to the public extension
interface.
