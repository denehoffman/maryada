use std::ops::{Add, Div, Mul, Sub};

use maryada::{ComplexBox, ComplexDisk, EnclosureOps, Interval};
use num_complex::Complex64;
use pyo3::{
    create_exception,
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyComplex, PyInt},
};

use super::{IntervalConstructionError, PyInterval, add_method_aliases, binary64, scalar_argument};

create_exception!(
    maryada,
    ComplexDiskError,
    PyValueError,
    "Invalid disk parameters or an unavailable finite disk enclosure."
);

type BareBox = ComplexBox<Interval>;

/// A rectangular enclosure with independent real and imaginary intervals.
#[pyclass(eq, frozen, module = "maryada", name = "ComplexBox")]
#[derive(PartialEq)]
struct PyComplexBox(BareBox);

/// A closed complex disk with a binary64 center and radius.
#[pyclass(eq, frozen, module = "maryada", name = "ComplexDisk")]
#[derive(PartialEq)]
struct PyComplexDisk(ComplexDisk);

macro_rules! complex_methods {
    (
        $class:ident, $datum:ident;
        [$($(#[$doc:meta])* $method:ident => $kernel:ident),+ $(,)?];
        $($extra:tt)*
    ) => {
        #[pymethods]
        impl $class {
            $($extra)*

            #[classattr]
            const EMPTY: Self = Self($datum::EMPTY);
            #[classattr]
            const ENTIRE: Self = Self($datum::ENTIRE);
            #[classattr]
            const ZERO: Self = Self($datum::ZERO);
            #[classattr]
            const ONE: Self = Self($datum::ONE);
            #[classattr]
            const I: Self = Self($datum::I);

            /// Return the empty complex set.
            #[staticmethod]
            fn empty() -> Self {
                Self::EMPTY
            }

            /// Return the entire complex plane.
            #[staticmethod]
            fn entire() -> Self {
                Self::ENTIRE
            }

            /// Enclose a single finite complex point.
            #[staticmethod]
            fn point(value: &Bound<'_, PyAny>) -> PyResult<Self> {
                Self::from_point(point_argument(value)?)
            }

            /// Enclose the points exp(i * phase).
            #[staticmethod]
            fn cis(phase: &Bound<'_, PyAny>) -> PyResult<Self> {
                Ok(Self($datum::cis(real_interval(phase, "phase")?)))
            }

            /// Enclose the points given by the radius and phase intervals.
            #[staticmethod]
            fn from_polar(radius: &Bound<'_, PyAny>, phase: &Bound<'_, PyAny>) -> PyResult<Self> {
                Ok(Self($datum::from_polar(
                    real_interval(radius, "radius")?,
                    real_interval(phase, "phase")?,
                )))
            }

            /// Return whether this enclosure is empty.
            fn is_empty(&self) -> bool {
                self.0.is_empty()
            }

            /// Return whether this enclosure contains the entire complex plane.
            fn is_entire(&self) -> bool {
                self.0.is_entire()
            }

            /// Return whether this enclosure is bounded.
            fn is_bounded(&self) -> bool {
                self.0.is_bounded()
            }

            /// Return whether this enclosure contains exactly one point.
            fn is_singleton(&self) -> bool {
                self.0.is_singleton()
            }

            /// Return whether this enclosure is the singleton zero.
            fn is_zero(&self) -> bool {
                self.0.is_zero()
            }

            /// Return whether all represented values are real.
            fn is_real(&self) -> bool {
                self.0.is_real()
            }

            /// Return whether this enclosure is contained in the other enclosure.
            fn subset(&self, other: &Self) -> bool {
                self.0.subset(other.0)
            }

            /// Return whether this enclosure is contained in the interior of the other.
            fn interior(&self, other: &Self) -> bool {
                self.0.interior(other.0)
            }

            /// Return whether the enclosures have no common points.
            fn disjoint(&self, other: &Self) -> bool {
                self.0.disjoint(other.0)
            }

            /// Return whether the enclosures have any common points.
            fn intersects(&self, other: &Self) -> bool {
                self.0.intersects(other.0)
            }

            /// Return a real interval enclosing the complex modulus.
            fn norm(&self) -> PyInterval {
                PyInterval(self.0.norm())
            }

            /// Return a real interval enclosing the squared modulus.
            fn norm_sqr(&self) -> PyInterval {
                PyInterval(self.0.norm_sqr())
            }

            /// Enclose principal arguments in radians.
            fn arg(&self) -> PyInterval {
                PyInterval(self.0.arg())
            }

            /// Return separate real intervals for modulus and principal argument.
            fn to_polar(&self) -> (PyInterval, PyInterval) {
                let (magnitude, angle) = self.0.to_polar();
                (PyInterval(magnitude), PyInterval(angle))
            }

            /// Enclose integer powers with a signed 32-bit exponent.
            fn pown(&self, exponent: i32) -> Self {
                Self(self.0.pown(exponent))
            }

            /// Enclose general powers using the principal logarithm.
            fn pow(&self, exponent: &Self) -> Self {
                Self(self.0.pow(exponent.0))
            }

            /// Enclose self * multiplier + addend.
            fn mul_add(&self, multiplier: &Self, addend: &Self) -> Self {
                Self(self.0.mul_add(multiplier.0, addend.0))
            }

            $(
                $(#[$doc])*
                fn $method(&self) -> Self {
                    Self(self.0.$kernel())
                }
            )+

            fn __abs__(&self) -> PyInterval {
                PyInterval(self.0.abs())
            }

            fn __neg__(&self) -> Self {
                Self(-self.0)
            }

            fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
                if let Ok(interval) = value.cast::<Self>() {
                    return Ok(interval.get().0.subset(self.0));
                }
                Ok(point_number(value)?.is_some_and(|point| self.0.contains(point)))
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
                    return Bound::new(py, self.pown(exponent.extract()?)).map(Bound::into_any);
                }
                self.binary_op(exponent, $datum::pow)
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

        impl $class {
            fn binary_op<'py>(
                &self,
                other: &Bound<'py, PyAny>,
                operation: impl FnOnce($datum, $datum) -> $datum,
            ) -> PyResult<Bound<'py, PyAny>> {
                let py = other.py();
                let rhs = if let Ok(interval) = other.cast::<Self>() {
                    interval.get().0
                } else if let Some(interval) = Self::promote_operand(other)? {
                    interval.0
                } else {
                    return Ok(py.NotImplemented().into_bound(py));
                };
                Bound::new(py, Self(operation(self.0, rhs))).map(Bound::into_any)
            }
        }
    };
}

// These named operations are implemented by both Rust enclosure types.
macro_rules! bind_complex {
    ($class:ident, $datum:ident; $($extra:tt)*) => {
        complex_methods!($class, $datum; [
            /// Return the complex conjugate.
            conjugate => conj,
            /// Enclose the reciprocals of nonzero members.
            reciprocal => recip,
            /// Enclose the squares of individual members.
            square => sqr,
            /// Enclose the principal square root.
            sqrt => sqrt,
            /// Enclose the natural exponential.
            exp => exp,
            /// Enclose the base-two exponential.
            exp2 => exp2,
            /// Enclose the base-ten exponential.
            exp10 => exp10,
            /// Enclose the principal natural logarithm.
            log => log,
            /// Enclose the principal base-two logarithm.
            log2 => log2,
            /// Enclose the principal base-ten logarithm.
            log10 => log10,
            /// Enclose the complex sine.
            sin => sin,
            /// Enclose the complex cosine.
            cos => cos,
            /// Enclose the complex tangent.
            tan => tan,
            /// Enclose the principal inverse sine.
            asin => asin,
            /// Enclose the principal inverse cosine.
            acos => acos,
            /// Enclose the principal inverse tangent.
            atan => atan,
            /// Enclose the complex hyperbolic sine.
            sinh => sinh,
            /// Enclose the complex hyperbolic cosine.
            cosh => cosh,
            /// Enclose the complex hyperbolic tangent.
            tanh => tanh,
            /// Enclose the principal inverse hyperbolic sine.
            asinh => asinh,
            /// Enclose the principal inverse hyperbolic cosine.
            acosh => acosh,
            /// Enclose the principal inverse hyperbolic tangent.
            atanh => atanh,
        ]; $($extra)*);
    };
}

bind_complex!(PyComplexBox, BareBox;
    /// Construct a box from its real and imaginary interval components.
    #[new]
    fn new(real: &Bound<'_, PyAny>, imag: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self(BareBox::new(
            real_interval(real, "real")?,
            real_interval(imag, "imag")?,
        )))
    }

    /// The real component.
    #[getter]
    fn real(&self) -> PyInterval {
        PyInterval(self.0.re)
    }

    /// The imaginary component.
    #[getter]
    fn imag(&self) -> PyInterval {
        PyInterval(self.0.im)
    }

    /// The midpoint as a Python complex number.
    #[getter]
    fn midpoint(&self) -> Complex64 {
        self.0.mid()
    }

    /// The real and imaginary widths.
    #[getter]
    fn width(&self) -> (f64, f64) {
        let widths = self.0.wid();
        (widths.re, widths.im)
    }

    /// A radius enclosing the box about its midpoint.
    #[getter]
    fn radius(&self) -> f64 {
        self.0.rad()
    }

    /// An inner radius about the box midpoint.
    #[getter]
    fn inner_radius(&self) -> f64 {
        self.0.inner_rad()
    }

    /// The midpoint and the separate real and imaginary radii.
    #[getter]
    fn midpoint_radius(&self) -> (Complex64, Complex64) {
        self.0.mid_rad()
    }

    /// An upper bound on the distance between any two points.
    #[getter]
    fn diameter(&self) -> f64 {
        self.0.diameter()
    }

    /// The lower real and imaginary endpoints.
    #[getter]
    fn lower_corner(&self) -> Complex64 {
        self.0.lower_corner()
    }

    /// The upper real and imaginary endpoints.
    #[getter]
    fn upper_corner(&self) -> Complex64 {
        self.0.upper_corner()
    }

    /// The four corners of the rectangular box.
    #[getter]
    fn corners(&self) -> (Complex64, Complex64, Complex64, Complex64) {
        let [first, second, third, fourth] = self.0.corners();
        (first, second, third, fourth)
    }

    /// Return the rectangular intersection.
    fn intersection(&self, other: &Self) -> Self {
        Self(self.0.intersection(other.0))
    }

    /// Return the smallest box enclosing both inputs.
    fn hull(&self, other: &Self) -> Self {
        Self(self.0.convex_hull(other.0))
    }

    /// Inflate the real and imaginary intervals.
    fn inflate(&self, relative: &Bound<'_, PyAny>, absolute: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (relative, absolute) = super::inflation_amounts(relative, absolute)?;
        Ok(Self(self.0.inflate(relative, absolute)))
    }

    /// Return an enclosing disk. Proper unbounded boxes raise ComplexDiskError.
    fn to_disk(&self) -> PyResult<PyComplexDisk> {
        ComplexDisk::try_from(self.0).map(PyComplexDisk).map_err(disk_error)
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        if self.0 == BareBox::EMPTY {
            return Ok("ComplexBox.empty()".into());
        }
        if self.0 == BareBox::ENTIRE {
            return Ok("ComplexBox.entire()".into());
        }
        Ok(format!(
            "ComplexBox({}, {})",
            self.real().__repr__(py)?,
            self.imag().__repr__(py)?
        ))
    }
);

impl PyComplexBox {
    fn from_point(point: Complex64) -> PyResult<Self> {
        BareBox::try_from((point.re, point.im))
            .map(Self)
            .map_err(|error| IntervalConstructionError::new_err(error.to_string()))
    }

    fn promote_operand(value: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        if let Ok(interval) = value.cast::<PyInterval>() {
            return Ok(Some(Self(BareBox::from(interval.get().0))));
        }
        point_number(value)?.map(Self::from_point).transpose()
    }
}

bind_complex!(PyComplexDisk, ComplexDisk;
    /// Construct a disk with a finite center and nonnegative finite radius.
    #[new]
    fn new(center: &Bound<'_, PyAny>, radius: &Bound<'_, PyAny>) -> PyResult<Self> {
        let point = point_argument(center)?;
        ComplexDisk::try_new((point.re, point.im), scalar_argument(radius, "radius")?)
            .map(Self)
            .map_err(disk_error)
    }

    /// Enclose a real interval in a disk centered on the real axis.
    #[staticmethod]
    fn from_interval(value: &PyInterval) -> PyResult<Self> {
        ComplexDisk::try_from(value.0).map(Self).map_err(disk_error)
    }

    /// The finite center, or None for Empty and Entire.
    #[getter]
    fn center(&self) -> Option<Complex64> {
        self.0.center().map(|(re, im)| Complex64::new(re, im))
    }

    /// The radius, with infinity for Entire and None for Empty.
    #[getter]
    fn radius(&self) -> Option<f64> {
        self.0.radius()
    }

    /// Return the rectangular hull of this disk.
    fn to_box(&self) -> PyComplexBox {
        PyComplexBox(BareBox::from(self.0))
    }

    /// Try a smaller-radius enclosure of the product with a shifted center.
    fn mul_recentered(&self, other: &Self) -> Self {
        Self(self.0.mul_recentered(other.0))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        match (self.0.center(), self.0.radius()) {
            (Some((re, im)), Some(radius)) => {
                let center = PyComplex::from_doubles(py, re, im).repr()?;
                let radius = pyo3::types::PyFloat::new(py, radius).repr()?;
                Ok(format!("ComplexDisk({}, {})", center.to_str()?, radius.to_str()?))
            }
            (_, Some(_)) => Ok("ComplexDisk.entire()".into()),
            (_, None) => Ok("ComplexDisk.empty()".into()),
        }
    }
);

impl PyComplexDisk {
    fn from_point(point: Complex64) -> PyResult<Self> {
        ComplexDisk::try_from((point.re, point.im))
            .map(Self)
            .map_err(disk_error)
    }

    fn promote_operand(value: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        point_number(value)?.map(Self::from_point).transpose()
    }
}

fn point_number(value: &Bound<'_, PyAny>) -> PyResult<Option<Complex64>> {
    if let Ok(point) = value.cast::<PyComplex>() {
        return Ok(Some(Complex64::new(point.real(), point.imag())));
    }
    Ok(binary64(value)?.map(|real| Complex64::new(real, 0.0)))
}

fn point_argument(value: &Bound<'_, PyAny>) -> PyResult<Complex64> {
    point_number(value)?.ok_or_else(|| PyTypeError::new_err("expected a complex, float, or int"))
}

fn real_interval(value: &Bound<'_, PyAny>, argument: &str) -> PyResult<Interval> {
    if let Ok(interval) = value.cast::<PyInterval>() {
        return Ok(interval.get().0);
    }
    let value = scalar_argument(value, argument)?;
    Interval::try_new(value, value)
        .map_err(|error| IntervalConstructionError::new_err(error.to_string()))
}

fn disk_error(error: maryada::ComplexDiskError) -> PyErr {
    ComplexDiskError::new_err(error.to_string())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyComplexBox>()?;
    module.add_class::<PyComplexDisk>()?;
    let aliases = [
        ("conj", "conjugate"),
        ("sqr", "square"),
        ("recip", "reciprocal"),
        ("powi", "pown"),
    ];
    add_method_aliases(&module.py().get_type::<PyComplexBox>(), &aliases)?;
    add_method_aliases(&module.py().get_type::<PyComplexDisk>(), &aliases)?;
    module.add(
        "ComplexDiskError",
        module.py().get_type::<ComplexDiskError>(),
    )
}
