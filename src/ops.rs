use crate::{
    DecoratedInterval, Decoration, Interval, IntervalDatum,
    rounding::{self, Direction},
    signals::{Signal, SignalSink},
};

#[derive(Clone, Copy)]
struct BareResult {
    interval: Interval,
    local: Decoration,
}

fn unary<T: IntervalDatum>(x: T, kernel: fn(Interval) -> BareResult) -> T {
    if x.__is_nai() {
        return x.__unary_result(Interval::EMPTY, Decoration::Ill);
    }

    let result = kernel(x.__interval());

    x.__unary_result(result.interval, result.local)
}

fn binary<T: IntervalDatum>(x: T, y: T, kernel: fn(Interval, Interval) -> BareResult) -> T {
    if x.__is_nai() || y.__is_nai() {
        return x.__binary_result(y, Interval::EMPTY, Decoration::Ill);
    }

    let result = kernel(x.__interval(), y.__interval());

    x.__binary_result(y, result.interval, result.local)
}

fn ternary<T: IntervalDatum>(
    x: T,
    y: T,
    z: T,
    kernel: fn(Interval, Interval, Interval) -> BareResult,
) -> T {
    if x.__is_nai() || y.__is_nai() || z.__is_nai() {
        return x.__ternary_result(y, z, Interval::EMPTY, Decoration::Ill);
    }

    let result = kernel(x.__interval(), y.__interval(), z.__interval());

    x.__ternary_result(y, z, result.interval, result.local)
}

/// Constructs an interval datum with endpoints `inf` and `sup`.
#[must_use]
pub fn new<T: IntervalDatum>(inf: f64, sup: f64) -> T {
    T::__from_nums(inf, sup)
}

/// Constructs the singleton interval containing `value`.
#[must_use]
pub fn singleton<T: IntervalDatum>(value: f64) -> T {
    T::__from_nums(value, value)
}

/// Returns the singleton interval containing zero.
#[must_use]
pub fn zero<T: IntervalDatum>() -> T {
    T::__zero()
}

/// Returns the singleton interval containing one.
#[must_use]
pub fn one<T: IntervalDatum>() -> T {
    T::__one()
}

// 6.7.1: interval constants.

/// Returns the empty interval.
#[must_use]
pub fn empty<T: IntervalDatum>() -> T {
    T::__empty()
}

/// Returns the interval containing every real number.
#[must_use]
pub fn entire<T: IntervalDatum>() -> T {
    T::__entire()
}

// 6.7.2: basic operations.

/// Computes the additive inverse of every value in `x`.
pub fn neg<T: IntervalDatum>(x: T) -> T {
    unary(x, neg_bare)
}

/// Encloses all pairwise sums of values from `x` and `y`.
pub fn add<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, add_bare)
}

/// Encloses all pairwise differences `x - y`.
pub fn sub<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, sub_bare)
}

/// Encloses all pairwise products of values from `x` and `y`.
pub fn mul<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, mul_bare)
}

/// Encloses all defined quotients `x / y`.
pub fn div<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, div_bare)
}

/// Encloses the reciprocals of the nonzero values in `x`.
pub fn recip<T: IntervalDatum>(x: T) -> T {
    unary(x, recip_bare)
}

/// Encloses the squares of all values in `x`.
pub fn sqr<T: IntervalDatum>(x: T) -> T {
    unary(x, sqr_bare)
}

/// Encloses the real square roots of the nonnegative part of `x`.
pub fn sqrt<T: IntervalDatum>(x: T) -> T {
    unary(x, sqrt_bare)
}

/// Encloses the fused expression `x * y + z` with one final rounding step.
pub fn fma<T: IntervalDatum>(x: T, y: T, z: T) -> T {
    ternary(x, y, z, fma_bare)
}

// Power functions.

/// Raises every value in `x` to the integer power `p`.
pub fn pown<T: IntervalDatum>(x: T, p: i32) -> T {
    if x.__is_nai() {
        return x.__unary_result(Interval::EMPTY, Decoration::Ill);
    }

    let result = pown_bare(x.__interval(), p);

    x.__unary_result(result.interval, result.local)
}

/// Encloses the real-valued power function for bases in `x` and exponents in `y`.
pub fn pow<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, pow_bare)
}

/// Encloses `e^x` over the input interval.
pub fn exp<T: IntervalDatum>(x: T) -> T {
    unary(x, exp_bare)
}

/// Encloses `2^x` over the input interval.
pub fn exp2<T: IntervalDatum>(x: T) -> T {
    unary(x, exp2_bare)
}

/// Encloses `10^x` over the input interval.
pub fn exp10<T: IntervalDatum>(x: T) -> T {
    unary(x, exp10_bare)
}

/// Encloses the natural logarithm over the positive part of `x`.
pub fn log<T: IntervalDatum>(x: T) -> T {
    unary(x, log_bare)
}

/// Encloses the base-two logarithm over the positive part of `x`.
pub fn log2<T: IntervalDatum>(x: T) -> T {
    unary(x, log2_bare)
}

/// Encloses the base-ten logarithm over the positive part of `x`.
pub fn log10<T: IntervalDatum>(x: T) -> T {
    unary(x, log10_bare)
}

// Trigonometric functions.

/// Encloses the sine of every value in `x`.
pub fn sin<T: IntervalDatum>(x: T) -> T {
    unary(x, sin_bare)
}

/// Encloses the cosine of every value in `x`.
pub fn cos<T: IntervalDatum>(x: T) -> T {
    unary(x, cos_bare)
}

/// Encloses the defined tangent values over `x`.
pub fn tan<T: IntervalDatum>(x: T) -> T {
    unary(x, tan_bare)
}

/// Encloses the inverse sine over the part of `x` in `[-1, 1]`.
pub fn asin<T: IntervalDatum>(x: T) -> T {
    unary(x, asin_bare)
}

/// Encloses the inverse cosine over the part of `x` in `[-1, 1]`.
pub fn acos<T: IntervalDatum>(x: T) -> T {
    unary(x, acos_bare)
}

/// Encloses the inverse tangent of every value in `x`.
pub fn atan<T: IntervalDatum>(x: T) -> T {
    unary(x, atan_bare)
}

/// Encloses the two-argument angle `atan2(y, x)`.
pub fn atan2<T: IntervalDatum>(y: T, x: T) -> T {
    binary(y, x, atan2_bare)
}

// Hyperbolic functions.

/// Encloses the hyperbolic sine of every value in `x`.
pub fn sinh<T: IntervalDatum>(x: T) -> T {
    unary(x, sinh_bare)
}

/// Encloses the hyperbolic cosine of every value in `x`.
pub fn cosh<T: IntervalDatum>(x: T) -> T {
    unary(x, cosh_bare)
}

/// Encloses the hyperbolic tangent of every value in `x`.
pub fn tanh<T: IntervalDatum>(x: T) -> T {
    unary(x, tanh_bare)
}

/// Encloses the inverse hyperbolic sine of every value in `x`.
pub fn asinh<T: IntervalDatum>(x: T) -> T {
    unary(x, asinh_bare)
}

/// Encloses inverse hyperbolic cosine over the part of `x` at least one.
pub fn acosh<T: IntervalDatum>(x: T) -> T {
    unary(x, acosh_bare)
}

/// Encloses inverse hyperbolic tangent over the part of `x` in `(-1, 1)`.
pub fn atanh<T: IntervalDatum>(x: T) -> T {
    unary(x, atanh_bare)
}

// Integer functions.

/// Maps negative values to `-1`, zero to `0`, and positive values to `1`.
pub fn sign<T: IntervalDatum>(x: T) -> T {
    unary(x, sign_bare)
}

/// Encloses the ceiling of every value in `x`.
pub fn ceil<T: IntervalDatum>(x: T) -> T {
    unary(x, ceil_bare)
}

/// Encloses the floor of every value in `x`.
pub fn floor<T: IntervalDatum>(x: T) -> T {
    unary(x, floor_bare)
}

/// Encloses truncation toward zero for every value in `x`.
pub fn trunc<T: IntervalDatum>(x: T) -> T {
    unary(x, trunc_bare)
}

/// Encloses rounding to nearest integer with ties to even.
pub fn round_ties_to_even<T: IntervalDatum>(x: T) -> T {
    unary(x, round_ties_to_even_bare)
}

/// Encloses rounding to nearest integer with ties away from zero.
pub fn round_ties_to_away<T: IntervalDatum>(x: T) -> T {
    unary(x, round_ties_to_away_bare)
}

// Absmax functions.

/// Encloses the absolute value of every value in `x`.
pub fn abs<T: IntervalDatum>(x: T) -> T {
    unary(x, abs_bare)
}

/// Encloses the pointwise minimum of values from `x` and `y`.
pub fn min<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, min_bare)
}

/// Encloses the pointwise maximum of values from `x` and `y`.
pub fn max<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, max_bare)
}

// hypot for convenience:

/// Encloses `sqrt(x² + y²)` for values drawn from both intervals.
pub fn hypot<T: IntervalDatum>(x: T, y: T) -> T {
    sqrt(add(sqr(x), sqr(y)))
}

// 6.7.3: cancellative operations.

/// Computes the cancellative subtraction operation `x ⊖ y`.
pub fn cancel_minus<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, cancel_minus_bare)
}

/// Computes the cancellative addition operation `x ⊕ y`.
pub fn cancel_plus<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, cancel_plus_bare)
}

// 6.7.4: set operations.

/// Returns the set intersection of `x` and `y`.
pub fn intersection<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, intersection_bare)
}

/// Returns the smallest interval containing both `x` and `y`.
pub fn convex_hull<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, convex_hull_bare)
}

// 6.7.6: numeric functions.

/// Returns the lower endpoint, or `NaN` for `NaI`.
pub fn inf<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        x.__interval().inf_raw()
    }
}

/// Returns the upper endpoint, or `NaN` for `NaI`.
pub fn sup<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        x.__interval().sup_raw()
    }
}

/// Returns a representative midpoint, or `NaN` for an empty interval or `NaI`.
///
/// For finite bounded inputs this uses the overflow-safe binary64 midpoint,
/// rounded to nearest. Unbounded inputs follow the cases prescribed by IEEE
/// 1788.1 section 6.7.6.
pub fn mid<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        mid_bare(x.__interval())
    }
}

/// Returns the interval width rounded upward.
pub fn wid<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        wid_bare(x.__interval())
    }
}

/// Returns the radius rounded upward.
pub fn rad<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        rad_bare(x.__interval())
    }
}

/// Returns the radius rounded downward.
pub fn inner_rad<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        inner_rad_bare(x.__interval())
    }
}

/// Returns the magnitude, the greatest absolute value in `x`.
pub fn mag<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        mag_bare(x.__interval())
    }
}

/// Returns the mignitude, the least absolute value in `x`.
pub fn mig<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        mig_bare(x.__interval())
    }
}

/// Returns a midpoint and an upward-rounded radius enclosing `x`.
pub fn mid_rad<T: IntervalDatum>(x: T) -> (f64, f64) {
    if x.__is_nai() {
        return (f64::NAN, f64::NAN);
    }

    let interval = x.__interval();
    let midpoint = mid_bare(interval);
    let radius = rad_from_mid_bare(interval, midpoint);

    (midpoint, radius)
}

// 6.7.7: boolean functions.

/// Returns whether `x` is the empty interval; `NaI` is not empty.
pub fn is_empty<T: IntervalDatum>(x: T) -> bool {
    !x.__is_nai() && x.__interval().is_empty_raw()
}

/// Returns whether `x` contains every real number.
pub fn is_entire<T: IntervalDatum>(x: T) -> bool {
    !x.__is_nai() && x.__interval().is_entire_raw()
}

/// Returns whether `x` and `y` denote the same set.
pub fn equal<T: IntervalDatum>(x: T, y: T) -> bool {
    if x.__is_nai() || y.__is_nai() {
        return false;
    }

    equal_bare(x.__interval(), y.__interval())
}

/// Returns whether every member of `x` is also a member of `y`.
pub fn subset<T: IntervalDatum>(x: T, y: T) -> bool {
    if x.__is_nai() || y.__is_nai() {
        return false;
    }

    subset_bare(x.__interval(), y.__interval())
}

/// Returns whether `x` is contained in the topological interior of `y`.
pub fn interior<T: IntervalDatum>(x: T, y: T) -> bool {
    if x.__is_nai() || y.__is_nai() {
        return false;
    }

    interior_bare(x.__interval(), y.__interval())
}

/// Returns whether `x` and `y` have no common members.
pub fn disjoint<T: IntervalDatum>(x: T, y: T) -> bool {
    if x.__is_nai() || y.__is_nai() {
        return false;
    }

    disjoint_bare(x.__interval(), y.__interval())
}

/// Returns whether a decorated interval is Not an Interval.
#[must_use]
pub fn is_nai(x: DecoratedInterval) -> bool {
    x.is_nai_raw()
}

// 6.7.8: operations on/with decorations.

/// Attaches the strongest valid decoration to a bare interval.
#[must_use]
pub const fn new_dec(x: Interval) -> DecoratedInterval {
    DecoratedInterval::new_dec_raw(x)
}

/// Extracts the bare interval, signaling and returning empty for `NaI`.
pub fn interval_part<S: SignalSink>(x: DecoratedInterval, signals: &mut S) -> Interval {
    if x.is_nai_raw() {
        signals.raise(Signal::IntvlPartOfNaI);
        Interval::EMPTY
    } else {
        x.interval_raw()
    }
}

/// Returns the decoration component of a decorated interval.
#[must_use]
pub const fn decoration_part(x: DecoratedInterval) -> Decoration {
    x.decoration_raw()
}

/// Attaches `decoration`, weakening it when required by the interval.
#[must_use]
pub fn set_dec(x: Interval, decoration: Decoration) -> DecoratedInterval {
    DecoratedInterval::set_dec_raw(x, decoration)
}

// Bare interval kernels.

fn neg_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }

    let interval = Interval::from_valid_bounds(-x.sup_raw(), -x.inf_raw());

    BareResult {
        local: continuous_unary_decoration(x, interval),
        interval,
    }
}

fn add_bare(x: Interval, y: Interval) -> BareResult {
    if x.is_empty_raw() || y.is_empty_raw() {
        return empty_result();
    }

    let interval = Interval::from_valid_bounds(
        rounding::add(x.inf_raw(), y.inf_raw(), Direction::Down),
        rounding::add(x.sup_raw(), y.sup_raw(), Direction::Up),
    );

    BareResult {
        local: continuous_binary_decoration(x, y, interval),
        interval,
    }
}

fn sub_bare(x: Interval, y: Interval) -> BareResult {
    if x.is_empty_raw() || y.is_empty_raw() {
        return empty_result();
    }

    let interval = Interval::from_valid_bounds(
        rounding::sub(x.inf_raw(), y.sup_raw(), Direction::Down),
        rounding::sub(x.sup_raw(), y.inf_raw(), Direction::Up),
    );

    BareResult {
        local: continuous_binary_decoration(x, y, interval),
        interval,
    }
}

fn mul_endpoint(x: f64, y: f64, direction: Direction) -> f64 {
    // Rounding mul intentionally does special things with 0*inf and 0*nan
    if x == 0.0 || y == 0.0 {
        return 0.0;
    }
    rounding::mul(x, y, direction)
}

fn mul_interval(x: Interval, y: Interval) -> Interval {
    if x.is_empty_raw() || y.is_empty_raw() {
        return Interval::EMPTY;
    }
    let x_inf = x.inf_raw();
    let x_sup = x.sup_raw();
    let y_inf = y.inf_raw();
    let y_sup = y.sup_raw();
    let lower = f64::min(
        f64::min(
            mul_endpoint(x_inf, y_inf, Direction::Down),
            mul_endpoint(x_inf, y_sup, Direction::Down),
        ),
        f64::min(
            mul_endpoint(x_sup, y_inf, Direction::Down),
            mul_endpoint(x_sup, y_sup, Direction::Down),
        ),
    );
    let upper = f64::max(
        f64::max(
            mul_endpoint(x_inf, y_inf, Direction::Up),
            mul_endpoint(x_inf, y_sup, Direction::Up),
        ),
        f64::max(
            mul_endpoint(x_sup, y_inf, Direction::Up),
            mul_endpoint(x_sup, y_sup, Direction::Up),
        ),
    );
    Interval::from_valid_bounds(lower, upper)
}

fn mul_bare(x: Interval, y: Interval) -> BareResult {
    if x.is_empty_raw() || y.is_empty_raw() {
        return empty_result();
    }
    let interval = mul_interval(x, y);
    BareResult {
        interval,
        local: continuous_binary_decoration(x, y, interval),
    }
}

fn recip_interval(x: Interval) -> Interval {
    if x.is_empty_raw() {
        return Interval::EMPTY;
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    if inf == 0.0 && sup == 0.0 {
        Interval::EMPTY
    } else if inf < 0.0 && sup > 0.0 {
        Interval::ENTIRE
    } else if inf == 0.0 {
        Interval::from_valid_bounds(rounding::recip(sup, Direction::Down), f64::INFINITY)
    } else if sup == 0.0 {
        Interval::from_valid_bounds(f64::NEG_INFINITY, rounding::recip(inf, Direction::Up))
    } else {
        Interval::from_valid_bounds(
            rounding::recip(sup, Direction::Down),
            rounding::recip(inf, Direction::Up),
        )
    }
}

fn div_interval(x: Interval, y: Interval) -> Interval {
    if x.is_empty_raw() || y.is_empty_raw() {
        return Interval::EMPTY;
    }
    let x_inf = x.inf_raw();
    let x_sup = x.sup_raw();
    let y_inf = y.inf_raw();
    let y_sup = y.sup_raw();
    if y_inf == 0.0 && y_sup == 0.0 {
        return Interval::EMPTY;
    }
    if x_inf == 0.0 && x_sup == 0.0 {
        return Interval::ZERO;
    }
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    for numerator in [x_inf, x_sup] {
        for denominator in [y_inf, y_sup] {
            if denominator == 0.0 {
                continue;
            }
            let down = rounding::div(numerator, denominator, Direction::Down);
            let up = rounding::div(numerator, denominator, Direction::Up);
            if !down.is_nan() {
                lower = lower.min(down);
            }
            if !up.is_nan() {
                upper = upper.max(up);
            }
        }
    }
    if y_inf == 0.0 || (y_inf < 0.0 && y_sup > 0.0) {
        if x_inf < 0.0 {
            lower = f64::NEG_INFINITY;
        }
        if x_sup > 0.0 {
            upper = f64::INFINITY;
        }
    }
    if y_sup == 0.0 || (y_inf < 0.0 && y_sup > 0.0) {
        if x_sup > 0.0 {
            lower = f64::NEG_INFINITY;
        }
        if x_inf < 0.0 {
            upper = f64::INFINITY;
        }
    }
    Interval::from_valid_bounds(lower, upper)
}

fn div_bare(x: Interval, y: Interval) -> BareResult {
    if x.is_empty_raw() || y.is_empty_raw() {
        return empty_result();
    }
    let interval = div_interval(x, y);
    let local = if y.contains_raw(0.0) {
        Decoration::Trv
    } else {
        continuous_binary_decoration(x, y, interval)
    };
    BareResult { interval, local }
}

fn recip_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval = recip_interval(x);
    let local = if x.contains_raw(0.0) {
        Decoration::Trv
    } else {
        continuous_unary_decoration(x, interval)
    };
    BareResult { interval, local }
}

fn sqr_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let interval = if inf >= 0.0 {
        Interval::from_valid_bounds(
            rounding::sqr(inf, Direction::Down),
            rounding::sqr(sup, Direction::Up),
        )
    } else if sup <= 0.0 {
        Interval::from_valid_bounds(
            rounding::sqr(sup, Direction::Down),
            rounding::sqr(inf, Direction::Up),
        )
    } else {
        let inf_squared = rounding::sqr(inf, Direction::Up);
        let sup_squared = rounding::sqr(sup, Direction::Up);
        Interval::from_valid_bounds(-0.0, f64::max(inf_squared, sup_squared))
    };
    BareResult {
        interval,
        local: continuous_unary_decoration(x, interval),
    }
}

fn sqrt_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let interval = if sup < 0.0 {
        Interval::EMPTY
    } else {
        let lower = if inf <= 0.0 {
            -0.0
        } else {
            rounding::sqrt(inf, Direction::Down)
        };
        let upper = rounding::sqrt(sup, Direction::Up);
        Interval::from_valid_bounds(lower, upper)
    };
    let local = if inf < 0.0 {
        Decoration::Trv
    } else {
        continuous_unary_decoration(x, interval)
    };
    BareResult { interval, local }
}

fn fma_endpoint(x: f64, y: f64, z: f64, direction: Direction) -> Option<f64> {
    if x == 0.0 || y == 0.0 {
        return Some(z);
    }
    let value = rounding::fma(x, y, z, direction);
    // We can still get NaN here from subtracting infinity from infinity
    if value.is_nan() { None } else { Some(value) }
}

fn fma_interval(x: Interval, y: Interval, z: Interval) -> Interval {
    let x_bounds = [x.inf_raw(), x.sup_raw()];
    let y_bounds = [y.inf_raw(), y.sup_raw()];
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    let mut found_lower = false;
    let mut found_upper = false;
    for x_bound in x_bounds {
        for y_bound in y_bounds {
            if let Some(value) = fma_endpoint(x_bound, y_bound, z.inf_raw(), Direction::Down) {
                if !found_lower || value < lower {
                    lower = value;
                }
                found_lower = true;
            }
            if let Some(value) = fma_endpoint(x_bound, y_bound, z.sup_raw(), Direction::Up) {
                if !found_upper || value > upper {
                    upper = value;
                }
                found_upper = true;
            }
        }
    }
    debug_assert!(found_lower);
    debug_assert!(found_upper);
    Interval::from_valid_bounds(lower, upper)
}

fn fma_bare(x: Interval, y: Interval, z: Interval) -> BareResult {
    if x.is_empty_raw() || y.is_empty_raw() || z.is_empty_raw() {
        return empty_result();
    }
    let interval = fma_interval(x, y, z);
    BareResult {
        interval,
        local: continuous_ternary_decoration(x, y, z, interval),
    }
}

fn pown_bare(x: Interval, p: i32) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let odd = p & 1 != 0;
    let interval = if p == 0 {
        Interval::from_valid_bounds(1.0, 1.0)
    } else if p > 0 && odd {
        Interval::from_valid_bounds(
            rounding::pown(inf, p, Direction::Down),
            rounding::pown(sup, p, Direction::Up),
        )
    } else if p > 0 {
        if inf >= 0.0 {
            Interval::from_valid_bounds(
                rounding::pown(inf, p, Direction::Down),
                rounding::pown(sup, p, Direction::Up),
            )
        } else if sup <= 0.0 {
            Interval::from_valid_bounds(
                rounding::pown(sup, p, Direction::Down),
                rounding::pown(inf, p, Direction::Up),
            )
        } else {
            let inf_power = rounding::pown(inf, p, Direction::Up);
            let sup_power = rounding::pown(sup, p, Direction::Up);
            Interval::from_valid_bounds(-0.0, f64::max(inf_power, sup_power))
        }
    } else if inf == 0.0 && sup == 0.0 {
        Interval::EMPTY
    } else if odd {
        if inf < 0.0 && sup > 0.0 {
            Interval::ENTIRE
        } else if inf == 0.0 {
            Interval::from_valid_bounds(rounding::pown(sup, p, Direction::Down), f64::INFINITY)
        } else if sup == 0.0 {
            Interval::from_valid_bounds(f64::NEG_INFINITY, rounding::pown(inf, p, Direction::Up))
        } else {
            Interval::from_valid_bounds(
                rounding::pown(sup, p, Direction::Down),
                rounding::pown(inf, p, Direction::Up),
            )
        }
    } else if inf < 0.0 && sup > 0.0 {
        let inf_power = rounding::pown(inf, p, Direction::Down);
        let sup_power = rounding::pown(sup, p, Direction::Down);
        Interval::from_valid_bounds(f64::min(inf_power, sup_power), f64::INFINITY)
    } else if inf == 0.0 {
        Interval::from_valid_bounds(rounding::pown(sup, p, Direction::Down), f64::INFINITY)
    } else if sup == 0.0 {
        Interval::from_valid_bounds(rounding::pown(inf, p, Direction::Down), f64::INFINITY)
    } else if inf > 0.0 {
        Interval::from_valid_bounds(
            rounding::pown(sup, p, Direction::Down),
            rounding::pown(inf, p, Direction::Up),
        )
    } else {
        Interval::from_valid_bounds(
            rounding::pown(inf, p, Direction::Down),
            rounding::pown(sup, p, Direction::Up),
        )
    };
    let local = if p < 0 && x.contains_raw(0.0) {
        Decoration::Trv
    } else {
        continuous_unary_decoration(x, interval)
    };
    BareResult { interval, local }
}

fn pow_bare(x: Interval, y: Interval) -> BareResult {
    if x.is_empty_raw() || y.is_empty_raw() {
        return empty_result();
    }
    let x_inf = x.inf_raw();
    let x_sup = x.sup_raw();
    let y_inf = y.inf_raw();
    let y_sup = y.sup_raw();
    let interval = if x_sup < 0.0 || (x_sup == 0.0 && y_sup <= 0.0) {
        Interval::EMPTY
    } else if x_sup == 0.0 {
        Interval::ZERO
    } else {
        let domain_x = Interval::from_valid_bounds(f64::max(x_inf, -0.0), x_sup);
        let logarithm = log_bare(domain_x).interval;
        let exponent = mul_interval(logarithm, y);
        exp_bare(exponent).interval
    };
    let domain_contains_box = x_inf > 0.0 || (x_inf == 0.0 && y_inf > 0.0);
    let local = if domain_contains_box {
        continuous_binary_decoration(x, y, interval)
    } else {
        Decoration::Trv
    };
    BareResult { interval, local }
}

fn exp_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval = Interval::from_valid_bounds(
        rounding::exp(x.inf_raw(), Direction::Down),
        rounding::exp(x.sup_raw(), Direction::Up),
    );
    BareResult {
        interval,
        local: continuous_unary_decoration(x, interval),
    }
}

fn exp2_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval = Interval::from_valid_bounds(
        rounding::exp2(x.inf_raw(), Direction::Down),
        rounding::exp2(x.sup_raw(), Direction::Up),
    );
    BareResult {
        interval,
        local: continuous_unary_decoration(x, interval),
    }
}

fn exp10_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval = Interval::from_valid_bounds(
        rounding::exp10(x.inf_raw(), Direction::Down),
        rounding::exp10(x.sup_raw(), Direction::Up),
    );
    BareResult {
        interval,
        local: continuous_unary_decoration(x, interval),
    }
}

fn log_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let interval = if sup <= 0.0 {
        Interval::EMPTY
    } else {
        let lower = if inf <= 0.0 {
            f64::NEG_INFINITY
        } else {
            rounding::log(inf, Direction::Down)
        };
        Interval::from_valid_bounds(lower, rounding::log(sup, Direction::Up))
    };
    let local = if inf <= 0.0 {
        Decoration::Trv
    } else {
        continuous_unary_decoration(x, interval)
    };
    BareResult { interval, local }
}

fn log2_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let interval = if sup <= 0.0 {
        Interval::EMPTY
    } else {
        let lower = if inf <= 0.0 {
            f64::NEG_INFINITY
        } else {
            rounding::log2(inf, Direction::Down)
        };
        Interval::from_valid_bounds(lower, rounding::log2(sup, Direction::Up))
    };
    let local = if inf <= 0.0 {
        Decoration::Trv
    } else {
        continuous_unary_decoration(x, interval)
    };
    BareResult { interval, local }
}

fn log10_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let interval = if sup <= 0.0 {
        Interval::EMPTY
    } else {
        let lower = if inf <= 0.0 {
            f64::NEG_INFINITY
        } else {
            rounding::log10(inf, Direction::Down)
        };
        Interval::from_valid_bounds(lower, rounding::log10(sup, Direction::Up))
    };
    let local = if inf <= 0.0 {
        Decoration::Trv
    } else {
        continuous_unary_decoration(x, interval)
    };
    BareResult { interval, local }
}

fn sin_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let lower = if rounding::contains_sin_minimum(inf, sup) {
        -1.0
    } else {
        f64::min(
            rounding::sin(inf, Direction::Down),
            rounding::sin(sup, Direction::Down),
        )
    };
    let upper = if rounding::contains_sin_maximum(inf, sup) {
        1.0
    } else {
        f64::max(
            rounding::sin(inf, Direction::Up),
            rounding::sin(sup, Direction::Up),
        )
    };
    let interval = Interval::from_valid_bounds(lower, upper);
    BareResult {
        interval,
        local: continuous_unary_decoration(x, interval),
    }
}

fn cos_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let lower = if rounding::contains_cos_minimum(inf, sup) {
        -1.0
    } else {
        f64::min(
            rounding::cos(inf, Direction::Down),
            rounding::cos(sup, Direction::Down),
        )
    };
    let upper = if rounding::contains_cos_maximum(inf, sup) {
        1.0
    } else {
        f64::max(
            rounding::cos(inf, Direction::Up),
            rounding::cos(sup, Direction::Up),
        )
    };
    let interval = Interval::from_valid_bounds(lower, upper);
    BareResult {
        interval,
        local: continuous_unary_decoration(x, interval),
    }
}

fn tan_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    if rounding::contains_tan_pole(inf, sup) {
        return BareResult {
            interval: Interval::ENTIRE,
            local: Decoration::Trv,
        };
    }
    let interval = Interval::from_valid_bounds(
        rounding::tan(inf, Direction::Down),
        rounding::tan(sup, Direction::Up),
    );
    BareResult {
        interval,
        local: continuous_unary_decoration(x, interval),
    }
}

fn asin_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let interval = if sup < -1.0 || inf > 1.0 {
        Interval::EMPTY
    } else {
        Interval::from_valid_bounds(
            rounding::asin(f64::max(inf, -1.0), Direction::Down),
            rounding::asin(f64::min(sup, 1.0), Direction::Up),
        )
    };
    let local = if inf < -1.0 || sup > 1.0 {
        Decoration::Trv
    } else {
        continuous_unary_decoration(x, interval)
    };
    BareResult { interval, local }
}

fn acos_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let interval = if sup < -1.0 || inf > 1.0 {
        Interval::EMPTY
    } else {
        Interval::from_valid_bounds(
            rounding::acos(f64::min(sup, 1.0), Direction::Down),
            rounding::acos(f64::max(inf, -1.0), Direction::Up),
        )
    };
    let local = if inf < -1.0 || sup > 1.0 {
        Decoration::Trv
    } else {
        continuous_unary_decoration(x, interval)
    };
    BareResult { interval, local }
}

fn atan_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval = Interval::from_valid_bounds(
        rounding::atan(x.inf_raw(), Direction::Down),
        rounding::atan(x.sup_raw(), Direction::Up),
    );
    BareResult {
        interval,
        local: continuous_unary_decoration(x, interval),
    }
}

fn atan2_bare(y: Interval, x: Interval) -> BareResult {
    if x.is_empty_raw() || y.is_empty_raw() {
        return empty_result();
    }
    let x_inf = x.inf_raw();
    let x_sup = x.sup_raw();
    let y_inf = y.inf_raw();
    let y_sup = y.sup_raw();
    let both_zero = x_inf == 0.0 && x_sup == 0.0 && y_inf == 0.0 && y_sup == 0.0;
    let interval = if both_zero {
        Interval::EMPTY
    } else if x_inf < 0.0 && y_inf < 0.0 && y_sup >= 0.0 {
        let pi_upper = core::f64::consts::PI.next_up();
        Interval::from_valid_bounds(-pi_upper, pi_upper)
    } else {
        let mut lower = f64::INFINITY;
        let mut upper = f64::NEG_INFINITY;
        for y_bound in [y_inf, y_sup] {
            for x_bound in [x_inf, x_sup] {
                if x_bound == 0.0 && y_bound == 0.0 {
                    continue;
                }
                let y_bound = if y_bound == 0.0 { 0.0 } else { y_bound };
                lower = f64::min(lower, rounding::atan2(y_bound, x_bound, Direction::Down));
                upper = f64::max(upper, rounding::atan2(y_bound, x_bound, Direction::Up));
            }
        }
        Interval::from_valid_bounds(lower, upper)
    };
    let local = if y.contains_raw(0.0) && x_inf <= 0.0 {
        Decoration::Trv
    } else {
        continuous_binary_decoration(y, x, interval)
    };
    BareResult { interval, local }
}

fn sinh_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval = Interval::from_valid_bounds(
        rounding::sinh(x.inf_raw(), Direction::Down),
        rounding::sinh(x.sup_raw(), Direction::Up),
    );
    BareResult {
        interval,
        local: continuous_unary_decoration(x, interval),
    }
}

fn cosh_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let lower = if x.contains_raw(0.0) {
        1.0
    } else if sup < 0.0 {
        rounding::cosh(sup, Direction::Down)
    } else {
        rounding::cosh(inf, Direction::Down)
    };
    let upper = if -inf > sup {
        rounding::cosh(inf, Direction::Up)
    } else {
        rounding::cosh(sup, Direction::Up)
    };
    let interval = Interval::from_valid_bounds(lower, upper);
    BareResult {
        interval,
        local: continuous_unary_decoration(x, interval),
    }
}

fn tanh_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval = Interval::from_valid_bounds(
        rounding::tanh(x.inf_raw(), Direction::Down),
        rounding::tanh(x.sup_raw(), Direction::Up),
    );
    BareResult {
        interval,
        local: continuous_unary_decoration(x, interval),
    }
}

fn asinh_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval = Interval::from_valid_bounds(
        rounding::asinh(x.inf_raw(), Direction::Down),
        rounding::asinh(x.sup_raw(), Direction::Up),
    );
    BareResult {
        interval,
        local: continuous_unary_decoration(x, interval),
    }
}

fn acosh_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let interval = if sup < 1.0 {
        Interval::EMPTY
    } else {
        let lower = if inf <= 1.0 {
            -0.0
        } else {
            rounding::acosh(inf, Direction::Down)
        };
        Interval::from_valid_bounds(lower, rounding::acosh(sup, Direction::Up))
    };
    let local = if inf < 1.0 {
        Decoration::Trv
    } else {
        continuous_unary_decoration(x, interval)
    };
    BareResult { interval, local }
}

fn atanh_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let interval = if sup <= -1.0 || inf >= 1.0 {
        Interval::EMPTY
    } else {
        let lower = if inf <= -1.0 {
            f64::NEG_INFINITY
        } else {
            rounding::atanh(inf, Direction::Down)
        };
        let upper = if sup >= 1.0 {
            f64::INFINITY
        } else {
            rounding::atanh(sup, Direction::Up)
        };
        Interval::from_valid_bounds(lower, upper)
    };
    let local = if inf <= -1.0 || sup >= 1.0 {
        Decoration::Trv
    } else {
        continuous_unary_decoration(x, interval)
    };
    BareResult { interval, local }
}

fn sign_bare(x: Interval) -> BareResult {
    fn sign_value(value: f64) -> f64 {
        if value < 0.0 {
            -1.0
        } else if value > 0.0 {
            1.0
        } else {
            0.0
        }
    }
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval = Interval::from_valid_bounds(sign_value(x.inf_raw()), sign_value(x.sup_raw()));
    BareResult {
        interval,
        local: integer_function_decoration(x, interval, sign_discontinuity(x)),
    }
}

fn ceil_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval = Interval::from_valid_bounds(libm::ceil(x.inf_raw()), libm::ceil(x.sup_raw()));
    BareResult {
        interval,
        local: integer_function_decoration(x, interval, ceil_discontinuity(x)),
    }
}

fn floor_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval = Interval::from_valid_bounds(libm::floor(x.inf_raw()), libm::floor(x.sup_raw()));
    BareResult {
        interval,
        local: integer_function_decoration(x, interval, floor_discontinuity(x)),
    }
}

fn trunc_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval = Interval::from_valid_bounds(libm::trunc(x.inf_raw()), libm::trunc(x.sup_raw()));
    BareResult {
        interval,
        local: integer_function_decoration(x, interval, trunc_discontinuity(x)),
    }
}

fn round_ties_to_even_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval =
        Interval::from_valid_bounds(libm::roundeven(x.inf_raw()), libm::roundeven(x.sup_raw()));
    BareResult {
        interval,
        local: integer_function_decoration(x, interval, round_ties_to_even_discontinuity(x)),
    }
}

fn round_ties_to_away_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval = Interval::from_valid_bounds(libm::round(x.inf_raw()), libm::round(x.sup_raw()));
    BareResult {
        interval,
        local: integer_function_decoration(x, interval, round_ties_to_away_discontinuity(x)),
    }
}

fn abs_bare(x: Interval) -> BareResult {
    if x.is_empty_raw() {
        return empty_result();
    }
    let interval = if x.contains_raw(0.0) {
        let upper = abs_max(x.inf_raw(), x.sup_raw());
        Interval::from_valid_bounds(-0.0, upper)
    } else {
        let a = x.inf_raw().abs();
        let b = x.sup_raw().abs();
        Interval::from_valid_bounds(if a < b { a } else { b }, if a > b { a } else { b })
    };
    BareResult {
        local: continuous_unary_decoration(x, interval),
        interval,
    }
}

fn min_bare(x: Interval, y: Interval) -> BareResult {
    if x.is_empty_raw() || y.is_empty_raw() {
        return empty_result();
    }
    let interval = Interval::from_valid_bounds(
        if x.inf_raw() < y.inf_raw() {
            x.inf_raw()
        } else {
            y.inf_raw()
        },
        if x.sup_raw() < y.sup_raw() {
            x.sup_raw()
        } else {
            y.sup_raw()
        },
    );
    BareResult {
        local: continuous_binary_decoration(x, y, interval),
        interval,
    }
}

fn max_bare(x: Interval, y: Interval) -> BareResult {
    if x.is_empty_raw() || y.is_empty_raw() {
        return empty_result();
    }
    let interval = Interval::from_valid_bounds(
        if x.inf_raw() > y.inf_raw() {
            x.inf_raw()
        } else {
            y.inf_raw()
        },
        if x.sup_raw() > y.sup_raw() {
            x.sup_raw()
        } else {
            y.sup_raw()
        },
    );
    BareResult {
        local: continuous_binary_decoration(x, y, interval),
        interval,
    }
}

fn cancel_minus_bare(x: Interval, y: Interval) -> BareResult {
    let interval = cancel_minus_interval(x, y);
    BareResult {
        interval,
        local: Decoration::Trv,
    }
}

fn cancel_plus_bare(x: Interval, y: Interval) -> BareResult {
    let neg_y = if y.is_empty_raw() {
        Interval::EMPTY
    } else {
        Interval::from_valid_bounds(-y.sup_raw(), -y.inf_raw())
    };
    let interval = cancel_minus_interval(x, neg_y);
    BareResult {
        interval,
        local: Decoration::Trv,
    }
}

fn cancel_minus_interval(x: Interval, y: Interval) -> Interval {
    fn exact_difference_parts(sup: f64, inf: f64) -> (f64, f64) {
        let neg_inf = -inf;
        let difference = sup + neg_inf;
        debug_assert!(difference.is_finite());
        let neg_inf_rounded = difference - sup;
        let error = (sup - (difference - neg_inf_rounded)) + (neg_inf - neg_inf_rounded);
        (difference, error)
    }

    fn width_at_least(x: Interval, y: Interval) -> bool {
        let x_width = x.sup_raw() - x.inf_raw();
        let y_width = y.sup_raw() - y.inf_raw();

        if x_width == f64::INFINITY || y_width == f64::INFINITY {
            if x_width != y_width {
                return x_width == f64::INFINITY;
            }

            // Both unscaled widths overflow. Halving these large finite
            // endpoints is exact and brings both differences into range.
            let x_parts = exact_difference_parts(x.sup_raw() * 0.5, x.inf_raw() * 0.5);
            let y_parts = exact_difference_parts(y.sup_raw() * 0.5, y.inf_raw() * 0.5);
            return x_parts.0 > y_parts.0 || (x_parts.0 == y_parts.0 && x_parts.1 >= y_parts.1);
        }

        let x_parts = exact_difference_parts(x.sup_raw(), x.inf_raw());
        let y_parts = exact_difference_parts(y.sup_raw(), y.inf_raw());
        x_parts.0 > y_parts.0 || (x_parts.0 == y_parts.0 && x_parts.1 >= y_parts.1)
    }

    if x.is_empty_raw() {
        return if y.is_empty_raw() || y.is_bounded_raw() {
            Interval::EMPTY
        } else {
            Interval::ENTIRE
        };
    }
    if y.is_empty_raw() || !x.is_bounded_raw() || !y.is_bounded_raw() {
        return Interval::ENTIRE;
    }
    if !width_at_least(x, y) {
        return Interval::ENTIRE;
    }

    Interval::from_valid_bounds(
        rounding::sub(x.inf_raw(), y.inf_raw(), Direction::Down),
        rounding::sub(x.sup_raw(), y.sup_raw(), Direction::Up),
    )
}

fn intersection_bare(x: Interval, y: Interval) -> BareResult {
    let interval = if x.is_empty_raw() || y.is_empty_raw() {
        Interval::EMPTY
    } else {
        let lower = if x.inf_raw() > y.inf_raw() {
            x.inf_raw()
        } else {
            y.inf_raw()
        };

        let upper = if x.sup_raw() < y.sup_raw() {
            x.sup_raw()
        } else {
            y.sup_raw()
        };

        if lower > upper {
            Interval::EMPTY
        } else {
            Interval::from_valid_bounds(lower, upper)
        }
    };
    BareResult {
        interval,
        local: Decoration::Trv,
    }
}

fn convex_hull_bare(x: Interval, y: Interval) -> BareResult {
    let interval = if x.is_empty_raw() {
        y
    } else if y.is_empty_raw() {
        x
    } else {
        Interval::from_valid_bounds(
            if x.inf_raw() < y.inf_raw() {
                x.inf_raw()
            } else {
                y.inf_raw()
            },
            if x.sup_raw() > y.sup_raw() {
                x.sup_raw()
            } else {
                y.sup_raw()
            },
        )
    };
    BareResult {
        interval,
        local: Decoration::Trv,
    }
}

// Numeric kernels.

fn mid_bare(x: Interval) -> f64 {
    if x.is_empty_raw() {
        return f64::NAN;
    }
    if x.is_entire_raw() {
        return 0.0;
    }
    if x.inf_raw() == f64::NEG_INFINITY {
        return -f64::MAX;
    }
    if x.sup_raw() == f64::INFINITY {
        return f64::MAX;
    }
    let midpoint = x.inf_raw().midpoint(x.sup_raw());
    if midpoint == 0.0 { 0.0 } else { midpoint }
}

fn wid_bare(x: Interval) -> f64 {
    if x.is_empty_raw() {
        return f64::NAN;
    }
    let value = rounding::sub(x.sup_raw(), x.inf_raw(), Direction::Up);
    // Note that -0.0 is considered equal to 0.0 so this normalizes
    if value == 0.0 { 0.0 } else { value }
}

fn rad_bare(x: Interval) -> f64 {
    let midpoint = mid_bare(x);
    rad_from_mid_bare(x, midpoint)
}

fn inner_rad_bare(x: Interval) -> f64 {
    let midpoint = mid_bare(x);
    inner_rad_from_mid_bare(x, midpoint)
}

fn rad_from_mid_bare(x: Interval, midpoint: f64) -> f64 {
    if x.is_empty_raw() {
        return f64::NAN;
    }
    rounding::radius(x.inf_raw(), x.sup_raw(), midpoint)
}

fn inner_rad_from_mid_bare(x: Interval, midpoint: f64) -> f64 {
    if x.is_empty_raw() {
        return f64::NAN;
    }
    rounding::inner_radius(x.inf_raw(), x.sup_raw(), midpoint)
}

fn mag_bare(x: Interval) -> f64 {
    if x.is_empty_raw() {
        return f64::NAN;
    }
    let value = abs_max(x.inf_raw(), x.sup_raw());
    // Note that -0.0 is considered equal to 0.0 so this normalizes
    if value == 0.0 { 0.0 } else { value }
}

fn mig_bare(x: Interval) -> f64 {
    if x.is_empty_raw() {
        return f64::NAN;
    }
    if x.contains_raw(0.0) {
        return 0.0;
    }
    let a = x.inf_raw().abs();
    let b = x.sup_raw().abs();
    let value = if a < b { a } else { b };
    // Note that -0.0 is considered equal to 0.0 so this normalizes
    if value == 0.0 { 0.0 } else { value }
}

// Boolean kernels.

fn equal_bare(x: Interval, y: Interval) -> bool {
    if x.is_empty_raw() || y.is_empty_raw() {
        return x.is_empty_raw() && y.is_empty_raw();
    }
    x.inf_raw() == y.inf_raw() && x.sup_raw() == y.sup_raw()
}

fn subset_bare(x: Interval, y: Interval) -> bool {
    if x.is_empty_raw() {
        return true;
    }
    if y.is_empty_raw() {
        return false;
    }
    y.inf_raw() <= x.inf_raw() && x.sup_raw() <= y.sup_raw()
}

fn interior_bare(x: Interval, y: Interval) -> bool {
    if x.is_empty_raw() {
        return true;
    }
    if y.is_empty_raw() {
        return false;
    }
    strict_or_same_infinity(y.inf_raw(), x.inf_raw())
        && strict_or_same_infinity(x.sup_raw(), y.sup_raw())
}

#[allow(clippy::suspicious_operation_groupings)]
fn disjoint_bare(x: Interval, y: Interval) -> bool {
    x.is_empty_raw() || y.is_empty_raw() || x.sup_raw() < y.inf_raw() || y.sup_raw() < x.inf_raw()
}

fn strict_or_same_infinity(left: f64, right: f64) -> bool {
    left < right || (left == right && left.is_infinite())
}

// Decoration helpers.

fn sign_discontinuity(x: Interval) -> Option<Decoration> {
    x.contains_raw(0.0)
        .then_some(if x.inf_raw() == x.sup_raw() {
            Decoration::Dac
        } else {
            Decoration::Def
        })
}

fn ceil_discontinuity(x: Interval) -> Option<Decoration> {
    let first = libm::ceil(x.inf_raw());
    if first > x.sup_raw() {
        None
    } else if first < x.sup_raw() {
        Some(Decoration::Def)
    } else {
        Some(Decoration::Dac)
    }
}

fn floor_discontinuity(x: Interval) -> Option<Decoration> {
    let last = libm::floor(x.sup_raw());
    if last < x.inf_raw() {
        None
    } else if x.inf_raw() < last {
        Some(Decoration::Def)
    } else {
        Some(Decoration::Dac)
    }
}

fn trunc_discontinuity(x: Interval) -> Option<Decoration> {
    let first = libm::ceil(x.inf_raw());
    let last = libm::floor(x.sup_raw());
    let contains_negative = first <= last && first < 0.0;
    let contains_positive = first <= last && last > 0.0;
    if (contains_negative && first < x.sup_raw()) || (contains_positive && x.inf_raw() < last) {
        Some(Decoration::Def)
    } else if contains_negative || contains_positive {
        Some(Decoration::Dac)
    } else {
        None
    }
}

fn is_half_integer(value: f64) -> bool {
    // 4_503_599_627_370_496 = 2^52 is the point where integers are no longer consecutive in f64s
    value.is_finite() && value.abs() < 4_503_599_627_370_496.0 && libm::floor(value) + 0.5 == value
}

fn has_interior_half_integer(x: Interval) -> bool {
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let width = sup - inf;
    if !inf.is_finite() || !sup.is_finite() || width > 1.0 {
        return true;
    }
    if width == 1.0 && !(is_half_integer(inf) && is_half_integer(sup)) {
        return true;
    }
    let first = libm::ceil(inf - 0.5) + 0.5;
    inf < first && first < sup
}

fn round_ties_to_even_discontinuity(x: Interval) -> Option<Decoration> {
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let lower_tie = is_half_integer(inf);
    let upper_tie = is_half_integer(sup);
    if !lower_tie && !upper_tie && !has_interior_half_integer(x) {
        return None;
    }
    if inf == sup {
        return Some(Decoration::Dac);
    }
    if has_interior_half_integer(x) {
        return Some(Decoration::Def);
    }
    // At k + 1/2, ties-to-even jumps to the right when k is even and
    // from the left when k is odd.
    let lower_jumps_right = lower_tie && libm::fmod(libm::floor(inf), 2.0) == 0.0;
    let upper_jumps_left = upper_tie && libm::fmod(libm::floor(sup), 2.0) != 0.0;
    Some(if lower_jumps_right || upper_jumps_left {
        Decoration::Def
    } else {
        Decoration::Dac
    })
}

fn round_ties_to_away_discontinuity(x: Interval) -> Option<Decoration> {
    let inf = x.inf_raw();
    let sup = x.sup_raw();
    let lower_tie = is_half_integer(inf);
    let upper_tie = is_half_integer(sup);
    if !lower_tie && !upper_tie && !has_interior_half_integer(x) {
        return None;
    }
    if inf == sup {
        return Some(Decoration::Dac);
    }
    if has_interior_half_integer(x) {
        return Some(Decoration::Def);
    }
    // Ties away from zero jump to the right on the negative half-line and
    // from the left on the positive half-line.
    Some(if (lower_tie && inf < 0.0) || (upper_tie && sup > 0.0) {
        Decoration::Def
    } else {
        Decoration::Dac
    })
}

fn integer_function_decoration(
    input: Interval,
    result: Interval,
    discontinuity: Option<Decoration>,
) -> Decoration {
    discontinuity.unwrap_or_else(|| continuous_unary_decoration(input, result))
}

const fn empty_result() -> BareResult {
    BareResult {
        interval: Interval::EMPTY,
        local: Decoration::Trv,
    }
}

const fn continuous_unary_decoration(input: Interval, result: Interval) -> Decoration {
    if input.is_empty_raw() {
        Decoration::Trv
    } else if input.is_bounded_raw() && result.is_bounded_raw() {
        Decoration::Com
    } else {
        Decoration::Dac
    }
}

const fn continuous_binary_decoration(x: Interval, y: Interval, result: Interval) -> Decoration {
    if x.is_empty_raw() || y.is_empty_raw() {
        Decoration::Trv
    } else if x.is_bounded_raw() && y.is_bounded_raw() && result.is_bounded_raw() {
        Decoration::Com
    } else {
        Decoration::Dac
    }
}

const fn continuous_ternary_decoration(
    x: Interval,
    y: Interval,
    z: Interval,
    result: Interval,
) -> Decoration {
    if x.is_empty_raw() || y.is_empty_raw() || z.is_empty_raw() {
        Decoration::Trv
    } else if x.is_bounded_raw()
        && y.is_bounded_raw()
        && z.is_bounded_raw()
        && result.is_bounded_raw()
    {
        Decoration::Com
    } else {
        Decoration::Dac
    }
}

fn abs_max(x: f64, y: f64) -> f64 {
    let x = x.abs();
    let y = y.abs();
    if x > y { x } else { y }
}
