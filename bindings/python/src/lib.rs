use std::{
    num::NonZeroU16,
    ops::{Add, Div, Mul, Sub},
};

use maryada::{
    DecimalIntervalFormat, EnclosureOps, Interval, IntervalOps, IntervalSet, Signal, SignalFlags,
    div_split, expm1, interval_to_text, log1p, recip_split, write_interval_decimal,
};
use pyo3::{
    create_exception,
    exceptions::{PyOverflowError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyFloat, PyInt, PyTuple},
};

create_exception!(
    maryada,
    IntervalConstructionError,
    PyValueError,
    "Numeric inputs do not define a valid interval."
);
create_exception!(
    maryada,
    IntervalParseError,
    PyValueError,
    "An interval literal is invalid or its endpoint order is ambiguous."
);

/// An immutable closed binary64 interval.
#[pyclass(eq, ord, frozen, module = "maryada", name = "Interval")]
#[derive(PartialEq, PartialOrd)]
struct PyInterval(Interval);

#[pymethods]
impl PyInterval {
    #[classattr]
    const ZERO: Self = Self(Interval::ZERO);
    #[classattr]
    const ONE: Self = Self(Interval::ONE);
    #[classattr]
    const EMPTY: Self = Self(Interval::EMPTY);
    #[classattr]
    const ENTIRE: Self = Self(Interval::ENTIRE);
    #[classattr]
    const PI: Self = Self(Interval::PI);
    #[classattr]
    const TAU: Self = Self(Interval::TAU);
    #[classattr]
    const E: Self = Self(Interval::E);
    #[classattr]
    const LN_2: Self = Self(Interval::LN_2);

    /// Construct an interval from two binary64 bounds.
    #[new]
    fn new(lower: &Bound<'_, PyAny>, upper: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::checked_bounds(
            scalar_argument(lower, "lower")?,
            scalar_argument(upper, "upper")?,
        )
    }

    /// Enclose a single finite binary64 value.
    #[staticmethod]
    fn point(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let value = scalar_argument(value, "value")?;
        Self::checked_bounds(value, value)
    }

    /// Construct an interval with outward-rounded midpoint ± radius bounds.
    #[staticmethod]
    fn from_mid_radius(midpoint: &Bound<'_, PyAny>, radius: &Bound<'_, PyAny>) -> PyResult<Self> {
        Interval::try_from_mid_rad(
            scalar_argument(midpoint, "midpoint")?,
            scalar_argument(radius, "radius")?,
        )
        .map(Self)
        .map_err(|error| IntervalConstructionError::new_err(error.to_string()))
    }

    /// Return the empty interval.
    #[staticmethod]
    fn empty() -> Self {
        Self(Interval::EMPTY)
    }

    /// Return the interval of all real numbers.
    #[staticmethod]
    fn entire() -> Self {
        Self(Interval::ENTIRE)
    }

    /// Parse an IEEE interval literal and round its bounds outward.
    #[staticmethod]
    fn parse(text: &str) -> PyResult<Self> {
        let (value, possibly_undefined) = Self::parse_checked(text)?;
        if possibly_undefined {
            return Err(IntervalParseError::new_err(
                "the exact order of the interval endpoints could not be determined",
            ));
        }
        Ok(value)
    }

    /// Return the enclosure and whether exact endpoint order was uncertain.
    #[staticmethod]
    fn parse_checked(text: &str) -> PyResult<(Self, bool)> {
        let (value, flags) = Interval::parse_checked(text)
            .map_err(|error| IntervalParseError::new_err(error.to_string()))?;
        Ok((
            Self(value),
            flags.contains(Signal::PossiblyUndefinedOperation),
        ))
    }

    /// Decode the exact IEEE binary encoding of a bare interval.
    #[staticmethod]
    #[pyo3(signature = (data, byteorder = "big"))]
    fn from_bytes(data: &Bound<'_, PyBytes>, byteorder: &str) -> PyResult<Self> {
        let mut signals = SignalFlags::NONE;
        let value = match byteorder {
            "big" => Interval::from_be_bytes(data.as_bytes(), &mut signals),
            "little" => Interval::from_le_bytes(data.as_bytes(), &mut signals),
            _ => return Err(PyValueError::new_err("byteorder must be 'big' or 'little'")),
        };
        if signals.is_empty() {
            Ok(Self(value))
        } else {
            Err(IntervalConstructionError::new_err(
                "invalid binary interval encoding",
            ))
        }
    }

    /// Enclose the sum of an iterable, rounding the endpoints once.
    #[staticmethod]
    fn sum(terms: &Bound<'_, PyAny>) -> PyResult<Self> {
        let terms = collect_intervals(terms)?;
        Ok(Self(maryada::sum(&terms)))
    }

    /// Enclose the dot product of two equal-length iterables.
    #[staticmethod]
    fn dot(left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> PyResult<Self> {
        let left = collect_intervals(left)?;
        let right = collect_intervals(right)?;
        maryada::dot(&left, &right)
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// The lower endpoint, or positive infinity for Empty.
    #[getter]
    fn lower(&self) -> f64 {
        self.0.inf()
    }

    /// The upper endpoint, or negative infinity for Empty.
    #[getter]
    fn upper(&self) -> f64 {
        self.0.sup()
    }

    /// Both endpoints in lower, upper order.
    #[getter]
    fn bounds(&self) -> (f64, f64) {
        self.0.bounds()
    }

    /// The representative midpoint.
    #[getter]
    fn midpoint(&self) -> f64 {
        self.0.mid()
    }

    /// An upper bound on the distance between the endpoints.
    #[getter]
    fn width(&self) -> f64 {
        self.0.wid()
    }

    /// An enclosing radius about the stored midpoint.
    #[getter]
    fn radius(&self) -> f64 {
        self.0.rad()
    }

    /// A radius about the midpoint whose points lie within the interval.
    #[getter]
    fn inner_radius(&self) -> f64 {
        self.0.inner_rad()
    }

    /// The midpoint and enclosing radius.
    #[getter]
    fn midpoint_radius(&self) -> (f64, f64) {
        self.0.mid_rad()
    }

    /// An upper bound on the absolute value of any member.
    #[getter]
    fn magnitude(&self) -> f64 {
        self.0.mag()
    }

    /// A lower bound on the absolute value of any member.
    #[getter]
    fn mignitude(&self) -> f64 {
        self.0.mig()
    }

    /// Return whether the interval contains no real numbers.
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Return whether the interval contains all real numbers.
    fn is_entire(&self) -> bool {
        self.0.is_entire()
    }

    /// Return whether both endpoints are finite and the interval is nonempty.
    fn is_bounded(&self) -> bool {
        self.0.is_bounded()
    }

    /// Return whether the interval contains exactly one point.
    fn is_singleton(&self) -> bool {
        self.0.is_singleton()
    }

    /// Return whether the value is NaI. Bare intervals always return False.
    fn is_nai(&self) -> bool {
        self.0.is_nai()
    }

    /// Return whether this interval is contained in the other interval.
    fn subset(&self, other: &Self) -> bool {
        self.0.subset(other.0)
    }

    /// Return whether this interval lies in the topological interior of the other.
    fn interior(&self, other: &Self) -> bool {
        self.0.interior(other.0)
    }

    /// Return whether the two intervals have no members in common.
    fn disjoint(&self, other: &Self) -> bool {
        self.0.disjoint(other.0)
    }

    /// Return whether the two intervals have members in common.
    fn intersects(&self, other: &Self) -> bool {
        self.0.intersects(other.0)
    }

    /// Return the intersection of the two intervals.
    fn intersection(&self, other: &Self) -> Self {
        Self(self.0.intersection(other.0))
    }

    /// Return the smallest interval containing both intervals.
    fn hull(&self, other: &Self) -> Self {
        Self(self.0.convex_hull(other.0))
    }

    /// Extend the hull to include a finite point.
    fn hull_value(&self, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let point = Self::point(value)?;
        Ok(self.hull(&point))
    }

    /// Split at the midpoint into two covering intervals.
    fn bisect(&self) -> (Self, Self) {
        let (left, right) = self.0.bisect();
        (Self(left), Self(right))
    }

    /// Inflate by relative and absolute amounts.
    fn inflate(&self, relative: &Bound<'_, PyAny>, absolute: &Bound<'_, PyAny>) -> PyResult<Self> {
        let relative = scalar_argument(relative, "relative")?;
        let absolute = scalar_argument(absolute, "absolute")?;
        if relative.is_nan() || relative < 0.0 || absolute.is_nan() || absolute < 0.0 {
            return Err(IntervalConstructionError::new_err(
                "inflation amounts must be nonnegative",
            ));
        }
        Ok(Self(self.0.inflate(relative, absolute)))
    }

    /// Compute cancellative subtraction as defined by IEEE 1788.1.
    fn cancel_minus(&self, other: &Self) -> Self {
        Self(self.0.cancel_minus(other.0))
    }

    /// Compute cancellative addition as defined by IEEE 1788.1.
    fn cancel_plus(&self, other: &Self) -> Self {
        Self(self.0.cancel_plus(other.0))
    }

    /// Enclose the pointwise minimum of the operands.
    fn min(&self, other: &Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// Enclose the pointwise maximum of the operands.
    fn max(&self, other: &Self) -> Self {
        Self(self.0.max(other.0))
    }

    /// Enclose the Euclidean norm of pairs of values.
    fn hypot(&self, other: &Self) -> Self {
        Self(self.0.hypot(other.0))
    }

    /// Enclose the two-argument angle atan2(self, x).
    fn atan2(&self, x: &Self) -> Self {
        Self(self.0.atan2(x.0))
    }

    /// Enclose self * multiplier + addend with fused endpoint rounding.
    fn mul_add(&self, multiplier: &Self, addend: &Self) -> Self {
        Self(self.0.mul_add(multiplier.0, addend.0))
    }

    /// Enclose reciprocals of the nonzero members.
    fn reciprocal(&self) -> Self {
        Self(self.0.recip())
    }

    /// Return the zero, one, or two components of the reciprocal enclosure.
    fn reciprocal_split<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        components(py, recip_split(self.0))
    }

    /// Return the zero, one, or two components of the quotient enclosure.
    fn div_split<'py>(&self, py: Python<'py>, divisor: &Self) -> PyResult<Bound<'py, PyTuple>> {
        components(py, div_split(self.0, divisor.0))
    }

    /// Enclose squares of individual members.
    fn square(&self) -> Self {
        Self(self.0.sqr())
    }

    /// Enclose an integer power with a signed 32-bit exponent.
    fn pown(&self, exponent: i32) -> Self {
        Self(self.0.pown(exponent))
    }

    /// Enclose general interval powers.
    fn pow(&self, exponent: &Self) -> Self {
        Self(self.0.pow(exponent.0))
    }

    /// Enclose square roots of the nonnegative members.
    fn sqrt(&self) -> Self {
        Self(self.0.sqrt())
    }

    /// Enclose the natural exponential.
    fn exp(&self) -> Self {
        Self(self.0.exp())
    }

    /// Enclose the exponential minus one.
    fn expm1(&self) -> Self {
        Self(expm1(self.0))
    }

    /// Enclose the base-two exponential.
    fn exp2(&self) -> Self {
        Self(self.0.exp2())
    }

    /// Enclose the base-ten exponential.
    fn exp10(&self) -> Self {
        Self(self.0.exp10())
    }

    /// Enclose natural logarithms of the positive members.
    fn log(&self) -> Self {
        Self(self.0.log())
    }

    /// Enclose the natural logarithm of one plus the input.
    fn log1p(&self) -> Self {
        Self(log1p(self.0))
    }

    /// Enclose the base-two logarithm.
    fn log2(&self) -> Self {
        Self(self.0.log2())
    }

    /// Enclose the base-ten logarithm.
    fn log10(&self) -> Self {
        Self(self.0.log10())
    }

    /// Enclose the sine.
    fn sin(&self) -> Self {
        Self(self.0.sin())
    }

    /// Enclose the cosine.
    fn cos(&self) -> Self {
        Self(self.0.cos())
    }

    /// Enclose the tangent where defined.
    fn tan(&self) -> Self {
        Self(self.0.tan())
    }

    /// Enclose the inverse sine.
    fn asin(&self) -> Self {
        Self(self.0.asin())
    }

    /// Enclose the inverse cosine.
    fn acos(&self) -> Self {
        Self(self.0.acos())
    }

    /// Enclose the inverse tangent.
    fn atan(&self) -> Self {
        Self(self.0.atan())
    }

    /// Enclose the hyperbolic sine.
    fn sinh(&self) -> Self {
        Self(self.0.sinh())
    }

    /// Enclose the hyperbolic cosine.
    fn cosh(&self) -> Self {
        Self(self.0.cosh())
    }

    /// Enclose the hyperbolic tangent.
    fn tanh(&self) -> Self {
        Self(self.0.tanh())
    }

    /// Enclose the inverse hyperbolic sine.
    fn asinh(&self) -> Self {
        Self(self.0.asinh())
    }

    /// Enclose the inverse hyperbolic cosine.
    fn acosh(&self) -> Self {
        Self(self.0.acosh())
    }

    /// Enclose the inverse hyperbolic tangent.
    fn atanh(&self) -> Self {
        Self(self.0.atanh())
    }

    /// Enclose the signs of the members.
    fn sign(&self) -> Self {
        Self(self.0.sign())
    }

    /// Enclose the pointwise ceilings.
    fn ceil(&self) -> Self {
        Self(self.0.ceil())
    }

    /// Enclose the pointwise floors.
    fn floor(&self) -> Self {
        Self(self.0.floor())
    }

    /// Enclose the pointwise truncations toward zero.
    fn trunc(&self) -> Self {
        Self(self.0.trunc())
    }

    /// Enclose the pointwise rounding to nearest integers, with ties to even.
    fn round_ties_to_even(&self) -> Self {
        Self(self.0.round_ties_to_even())
    }

    /// Enclose the pointwise rounding to nearest integers, with ties away from zero.
    fn round_ties_to_away(&self) -> Self {
        Self(self.0.round_ties_to_away())
    }

    /// Return an IEEE hexadecimal literal that preserves both endpoint bits.
    fn to_text(&self) -> PyResult<String> {
        let mut buffer = [0; 64];
        let length = interval_to_text(self.0, None, &mut buffer)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        std::str::from_utf8(&buffer[..length])
            .map(str::to_owned)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Format an outward-rounded decimal literal.
    #[pyo3(signature = (digits, *, notation = "fixed"))]
    fn to_decimal(&self, digits: u16, notation: &str) -> PyResult<String> {
        let format = match notation {
            "fixed" => DecimalIntervalFormat::Fixed {
                fractional_digits: digits,
            },
            "scientific" => DecimalIntervalFormat::Scientific {
                significant_digits: NonZeroU16::new(digits).ok_or_else(|| {
                    PyValueError::new_err("scientific notation needs at least one digit")
                })?,
            },
            "center_radius" => DecimalIntervalFormat::CenterRadius {
                fractional_digits: digits,
            },
            _ => return Err(PyValueError::new_err("unknown decimal notation")),
        };
        let mut text = String::new();
        write_interval_decimal(self.0, format, &mut text)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(text)
    }

    /// Encode both endpoints in the exact IEEE binary format.
    #[pyo3(signature = (byteorder = "big"))]
    fn to_bytes<'py>(&self, py: Python<'py>, byteorder: &str) -> PyResult<Bound<'py, PyBytes>> {
        let data = match byteorder {
            "big" => self.0.to_be_bytes(),
            "little" => self.0.to_le_bytes(),
            _ => return Err(PyValueError::new_err("byteorder must be 'big' or 'little'")),
        };
        Ok(PyBytes::new(py, &data))
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        if self.0.is_empty() {
            return Ok("Interval.empty()".into());
        }
        if self.0.is_entire() {
            return Ok("Interval.entire()".into());
        }
        let endpoint = |value| match value {
            f64::NEG_INFINITY => Ok("float('-inf')".into()),
            f64::INFINITY => Ok("float('inf')".into()),
            value => {
                // The constructor restores the canonical signed zeros.
                let value = if value == 0.0 { 0.0 } else { value };
                PyFloat::new(py, value).repr()?.to_str().map(str::to_owned)
            }
        };
        let (lower, upper) = self.0.bounds();
        Ok(format!(
            "Interval({}, {})",
            endpoint(lower)?,
            endpoint(upper)?
        ))
    }

    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(interval) = value.cast::<Self>() {
            return Ok(interval.get().subset(self));
        }
        if let Ok(number) = value.cast::<PyFloat>() {
            return Ok(self.0.contains(number.value()));
        }
        if self.0.is_empty() || !value.is_instance_of::<PyInt>() {
            return Ok(false);
        }
        // Compare in Python so an integer can be tested without rounding.
        let (lower, upper) = self.0.bounds();
        let py = value.py();
        Ok(PyFloat::new(py, lower).le(value)? && PyFloat::new(py, upper).ge(value)?)
    }

    fn __neg__(&self) -> Self {
        Self(-self.0)
    }

    fn __abs__(&self) -> Self {
        Self(self.0.abs())
    }

    fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary_op(other, Add::add)
    }

    fn __radd__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary_op(other, |left, right| right + left)
    }

    fn __sub__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary_op(other, Sub::sub)
    }

    fn __rsub__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary_op(other, |left, right| right - left)
    }

    fn __mul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary_op(other, Mul::mul)
    }

    fn __rmul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary_op(other, |left, right| right * left)
    }

    fn __truediv__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary_op(other, Div::div)
    }

    fn __rtruediv__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.binary_op(other, |left, right| right / left)
    }

    fn __pow__<'py>(
        &self,
        exponent: &Bound<'py, PyAny>,
        modulo: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = exponent.py();
        if modulo.is_some() {
            return Ok(py.NotImplemented().into_bound(py));
        }
        if exponent.is_instance_of::<PyInt>() {
            let value = self.pown(exponent.extract()?);
            return Bound::new(py, value).map(Bound::into_any);
        }
        self.binary_op(exponent, EnclosureOps::pow)
    }

    fn __rpow__<'py>(
        &self,
        base: &Bound<'py, PyAny>,
        modulo: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if modulo.is_some() {
            let py = base.py();
            return Ok(py.NotImplemented().into_bound(py));
        }
        self.binary_op(base, |exponent, base| base.pow(exponent))
    }
}

impl PyInterval {
    fn checked_bounds(lower: f64, upper: f64) -> PyResult<Self> {
        Interval::try_new(lower, upper)
            .map(Self)
            .map_err(|error| IntervalConstructionError::new_err(error.to_string()))
    }

    fn binary_op<'py>(
        &self,
        other: &Bound<'py, PyAny>,
        operation: impl FnOnce(Interval, Interval) -> Interval,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rhs = if let Ok(interval) = other.cast::<Self>() {
            interval.get().0
        } else if let Some(value) = binary64(other)? {
            Self::checked_bounds(value, value)?.0
        } else {
            return Ok(other.py().NotImplemented().into_bound(other.py()));
        };
        Bound::new(other.py(), Self(operation(self.0, rhs))).map(Bound::into_any)
    }
}

fn scalar_argument(value: &Bound<'_, PyAny>, argument: &str) -> PyResult<f64> {
    binary64(value)?
        .ok_or_else(|| PyTypeError::new_err(format!("{argument} must be a float or int")))
}

fn components(py: Python<'_>, set: IntervalSet) -> PyResult<Bound<'_, PyTuple>> {
    PyTuple::new(py, set.components().iter().copied().map(PyInterval))
}

fn collect_intervals(iterable: &Bound<'_, PyAny>) -> PyResult<Vec<Interval>> {
    iterable
        .try_iter()?
        .map(|item| Ok(item?.cast::<PyInterval>()?.get().0))
        .collect()
}

fn binary64(value: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    if let Ok(number) = value.cast::<PyFloat>() {
        return Ok(Some(number.value()));
    }
    if value.is_instance_of::<PyInt>() {
        let rounded: f64 = value.extract()?;
        // Python's comparison of a builtin float with an int checks the exact
        // integer value, including bits lost in conversion to binary64.
        if !PyFloat::new(value.py(), rounded).as_any().eq(value)? {
            return Err(PyOverflowError::new_err(
                "integer cannot be represented exactly as binary64",
            ));
        }
        return Ok(Some(rounded));
    }
    Ok(None)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyInterval>()?;
    module.add(
        "IntervalConstructionError",
        module.py().get_type::<IntervalConstructionError>(),
    )?;
    module.add(
        "IntervalParseError",
        module.py().get_type::<IntervalParseError>(),
    )
}
