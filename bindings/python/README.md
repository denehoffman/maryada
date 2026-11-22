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
Equality compares intervals as sets. Use `in` for point membership or interval
containment. For intervals, `a in b` is equivalent to `a.subset(b)`, including
when both intervals are equal or `a` is empty.

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
