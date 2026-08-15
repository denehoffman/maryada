//! A `no_std` binary64 real interval arithmetic library conforming to IEEE 1788.1.
//!
//! The rectangular complex interval extension is outside the scope of the
//! standard.
//!
//! maryada provides [`Interval`] and [`DecoratedInterval`] for outward-rounded
//! real interval arithmetic, `ComplexBox` for rectangular complex intervals
//! when the `complex` feature is enabled, and text and binary interchange operations. Most arithmetic functions are
//! generic over [`IntervalDatum`], so the same standards-oriented free-function
//! API works with bare and decorated intervals.
//!
//! [`IntervalOps`] offers the same operations in a chaining-friendly form:
//!
//! ```
//! use maryada::{Interval, IntervalOps};
//!
//! let x = Interval::new(1.0, 2.0);
//! let y = x.sqr();
//!
//! assert_eq!(y.bounds(), (1.0, 4.0));
//! ```
//!
//! Operations return enclosing intervals. Decorated operations additionally
//! propagate the weakest input or operation decoration. APIs that report IEEE
//! exception conditions accept a [`SignalSink`]; pass `&mut ()` to ignore them.

#![no_std]

#[cfg(feature = "complex")]
mod complex;
mod interchange;
#[cfg(feature = "linalg")]
mod linalg;
mod ops;
mod rounding;
mod signals;
mod text;
mod types;
mod ux;

#[cfg(feature = "complex")]
pub use complex::ComplexBox;
pub use interchange::*;
#[cfg(all(feature = "linalg", feature = "alloc"))]
pub use linalg::{DIntervalMatrix, DIntervalVector};
#[cfg(feature = "linalg")]
pub use linalg::{
    EpsilonInflation, GaussianElimination, HBR, IntervalMatrix, OIntervalMatrix, OIntervalVector,
    SIntervalMatrix, SIntervalVector, Solver,
};
pub use ops::*;
pub use signals::{Signal, SignalFlags, SignalSink};
pub use text::{ParseIntervalError, TextError, interval_to_text};
pub use types::{DecoratedInterval, Decoration, Interval, IntervalDatum, InvalidDecoration};
pub use ux::IntervalOps;

pub mod prelude {
    //! Common interval types for glob imports.

    pub use crate::{DecoratedInterval, Decoration, Interval, IntervalOps};
}
