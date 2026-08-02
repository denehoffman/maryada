//! A `no_std` binary64 interval arithmetic library based on IEEE 1788.1.
//!
//! maryada provides [`Interval`] and [`DecoratedInterval`] for outward-rounded
//! real interval arithmetic, [`ComplexBox`] for rectangular complex intervals,
//! and text and binary interchange operations. Most arithmetic functions are
//! generic over [`IntervalDatum`], so the same free-function API works with
//! bare and decorated intervals.
//!
//! The inherent methods offer the same operations in a chaining-friendly form:
//!
//! ```
//! use maryada::Interval;
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
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

mod complex;
mod interchange;
mod ops;
mod rounding;
mod signals;
mod text;
mod types;
mod ux;

pub use complex::ComplexBox;
pub use interchange::*;
pub use ops::*;
pub use signals::{Signal, SignalFlags, SignalSink};
pub use text::{ParseIntervalError, TextError, interval_to_text};
pub use types::{DecoratedInterval, Decoration, Interval, IntervalDatum, InvalidDecoration};

pub mod prelude {
    //! Common interval types for glob imports.

    pub use crate::types::{DecoratedInterval, Decoration, Interval};
}
