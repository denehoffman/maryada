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

// 6.7.1: interval constants.

pub fn empty<T: IntervalDatum>() -> T {
    T::__empty()
}

pub fn entire<T: IntervalDatum>() -> T {
    T::__entire()
}

// 6.7.2: basic operations.

pub fn neg<T: IntervalDatum>(x: T) -> T {
    unary(x, neg_bare)
}

pub fn add<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, add_bare)
}

pub fn sub<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, sub_bare)
}

pub fn mul<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, mul_bare)
}

pub fn div<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, div_bare)
}

pub fn recip<T: IntervalDatum>(x: T) -> T {
    unary(x, recip_bare)
}

pub fn sqr<T: IntervalDatum>(x: T) -> T {
    unary(x, sqr_bare)
}

pub fn sqrt<T: IntervalDatum>(x: T) -> T {
    unary(x, sqrt_bare)
}

pub fn fma<T: IntervalDatum>(x: T, y: T, z: T) -> T {
    ternary(x, y, z, fma_bare)
}

// Power functions.

pub fn pown<T: IntervalDatum>(x: T, p: i32) -> T {
    if x.__is_nai() {
        return x.__unary_result(Interval::EMPTY, Decoration::Ill);
    }

    let result = pown_bare(x.__interval(), p);

    x.__unary_result(result.interval, result.local)
}

pub fn pow<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, pow_bare)
}

pub fn exp<T: IntervalDatum>(x: T) -> T {
    unary(x, exp_bare)
}

pub fn exp2<T: IntervalDatum>(x: T) -> T {
    unary(x, exp2_bare)
}

pub fn exp10<T: IntervalDatum>(x: T) -> T {
    unary(x, exp10_bare)
}

pub fn log<T: IntervalDatum>(x: T) -> T {
    unary(x, log_bare)
}

pub fn log2<T: IntervalDatum>(x: T) -> T {
    unary(x, log2_bare)
}

pub fn log10<T: IntervalDatum>(x: T) -> T {
    unary(x, log10_bare)
}

// Trigonometric functions.

pub fn sin<T: IntervalDatum>(x: T) -> T {
    unary(x, sin_bare)
}

pub fn cos<T: IntervalDatum>(x: T) -> T {
    unary(x, cos_bare)
}

pub fn tan<T: IntervalDatum>(x: T) -> T {
    unary(x, tan_bare)
}

pub fn asin<T: IntervalDatum>(x: T) -> T {
    unary(x, asin_bare)
}

pub fn acos<T: IntervalDatum>(x: T) -> T {
    unary(x, acos_bare)
}

pub fn atan<T: IntervalDatum>(x: T) -> T {
    unary(x, atan_bare)
}

pub fn atan2<T: IntervalDatum>(y: T, x: T) -> T {
    binary(y, x, atan2_bare)
}

// Hyperbolic functions.

pub fn sinh<T: IntervalDatum>(x: T) -> T {
    unary(x, sinh_bare)
}

pub fn cosh<T: IntervalDatum>(x: T) -> T {
    unary(x, cosh_bare)
}

pub fn tanh<T: IntervalDatum>(x: T) -> T {
    unary(x, tanh_bare)
}

pub fn asinh<T: IntervalDatum>(x: T) -> T {
    unary(x, asinh_bare)
}

pub fn acosh<T: IntervalDatum>(x: T) -> T {
    unary(x, acosh_bare)
}

pub fn atanh<T: IntervalDatum>(x: T) -> T {
    unary(x, atanh_bare)
}

// Integer functions.

pub fn sign<T: IntervalDatum>(x: T) -> T {
    unary(x, sign_bare)
}

pub fn ceil<T: IntervalDatum>(x: T) -> T {
    unary(x, ceil_bare)
}

pub fn floor<T: IntervalDatum>(x: T) -> T {
    unary(x, floor_bare)
}

pub fn trunc<T: IntervalDatum>(x: T) -> T {
    unary(x, trunc_bare)
}

pub fn round_ties_to_even<T: IntervalDatum>(x: T) -> T {
    unary(x, round_ties_to_even_bare)
}

pub fn round_ties_to_away<T: IntervalDatum>(x: T) -> T {
    unary(x, round_ties_to_away_bare)
}

// Absmax functions.

pub fn abs<T: IntervalDatum>(x: T) -> T {
    unary(x, abs_bare)
}

pub fn min<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, min_bare)
}

pub fn max<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, max_bare)
}

// 6.7.3: cancellative operations.

pub fn cancel_minus<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, cancel_minus_bare)
}

pub fn cancel_plus<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, cancel_plus_bare)
}

// 6.7.4: set operations.

pub fn intersection<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, intersection_bare)
}

pub fn convex_hull<T: IntervalDatum>(x: T, y: T) -> T {
    binary(x, y, convex_hull_bare)
}

// 6.7.6: numeric functions.

pub fn inf<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        x.__interval().inf_raw()
    }
}

pub fn sup<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        x.__interval().sup_raw()
    }
}

pub fn mid<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        mid_bare(x.__interval())
    }
}

pub fn wid<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        wid_bare(x.__interval())
    }
}

pub fn rad<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        rad_bare(x.__interval())
    }
}

pub fn mag<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        mag_bare(x.__interval())
    }
}

pub fn mig<T: IntervalDatum>(x: T) -> f64 {
    if x.__is_nai() {
        f64::NAN
    } else {
        mig_bare(x.__interval())
    }
}

/// Recommended combined midpoint/radius operation.
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

pub fn is_empty<T: IntervalDatum>(x: T) -> bool {
    !x.__is_nai() && x.__interval().is_empty_raw()
}

pub fn is_entire<T: IntervalDatum>(x: T) -> bool {
    !x.__is_nai() && x.__interval().is_entire_raw()
}

pub fn equal<T: IntervalDatum>(x: T, y: T) -> bool {
    if x.__is_nai() || y.__is_nai() {
        return false;
    }

    equal_bare(x.__interval(), y.__interval())
}

pub fn subset<T: IntervalDatum>(x: T, y: T) -> bool {
    if x.__is_nai() || y.__is_nai() {
        return false;
    }

    subset_bare(x.__interval(), y.__interval())
}

pub fn interior<T: IntervalDatum>(x: T, y: T) -> bool {
    if x.__is_nai() || y.__is_nai() {
        return false;
    }

    interior_bare(x.__interval(), y.__interval())
}

pub fn disjoint<T: IntervalDatum>(x: T, y: T) -> bool {
    if x.__is_nai() || y.__is_nai() {
        return false;
    }

    disjoint_bare(x.__interval(), y.__interval())
}

pub fn is_nai(x: DecoratedInterval) -> bool {
    x.is_nai_raw()
}

// 6.7.8: operations on/with decorations.

pub const fn new_dec(x: Interval) -> DecoratedInterval {
    DecoratedInterval::new_dec_raw(x)
}

pub fn interval_part<S: SignalSink>(x: DecoratedInterval, signals: &mut S) -> Interval {
    if x.is_nai_raw() {
        signals.raise(Signal::IntvlPartOfNaI);
        Interval::EMPTY
    } else {
        x.interval_raw()
    }
}

pub const fn decoration_part(x: DecoratedInterval) -> Decoration {
    x.decoration_raw()
}

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

fn div_bare(x: Interval, y: Interval) -> BareResult {
    if x.is_empty_raw() || y.is_empty_raw() {
        return empty_result();
    }
    let interval = mul_interval(x, recip_interval(y));
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
    todo!()
}

fn pow_bare(x: Interval, y: Interval) -> BareResult {
    todo!()
}

fn exp_bare(x: Interval) -> BareResult {
    todo!()
}

fn exp2_bare(x: Interval) -> BareResult {
    todo!()
}

fn exp10_bare(x: Interval) -> BareResult {
    todo!()
}

fn log_bare(x: Interval) -> BareResult {
    todo!()
}

fn log2_bare(x: Interval) -> BareResult {
    todo!()
}

fn log10_bare(x: Interval) -> BareResult {
    todo!()
}

fn sin_bare(x: Interval) -> BareResult {
    // Endpoint evaluation alone is insufficient.
    // Include -1 or +1 when the interval contains an extremum.
    todo!()
}

fn cos_bare(x: Interval) -> BareResult {
    todo!()
}

fn tan_bare(x: Interval) -> BareResult {
    // Return Entire if the interval crosses a tangent pole.
    todo!()
}

fn asin_bare(x: Interval) -> BareResult {
    todo!()
}

fn acos_bare(x: Interval) -> BareResult {
    todo!()
}

fn atan_bare(x: Interval) -> BareResult {
    todo!()
}

fn atan2_bare(y: Interval, x: Interval) -> BareResult {
    // Must account for:
    //
    // - exclusion of (0,0);
    // - the branch discontinuity on y=0, x<0;
    // - range convention (-pi, pi].
    todo!()
}

fn sinh_bare(x: Interval) -> BareResult {
    todo!()
}

fn cosh_bare(x: Interval) -> BareResult {
    todo!()
}

fn tanh_bare(x: Interval) -> BareResult {
    todo!()
}

fn asinh_bare(x: Interval) -> BareResult {
    todo!()
}

fn acosh_bare(x: Interval) -> BareResult {
    todo!()
}

fn atanh_bare(x: Interval) -> BareResult {
    todo!()
}

fn sign_bare(x: Interval) -> BareResult {
    todo!()
}

fn ceil_bare(x: Interval) -> BareResult {
    todo!()
}

fn floor_bare(x: Interval) -> BareResult {
    todo!()
}

fn trunc_bare(x: Interval) -> BareResult {
    todo!()
}

fn round_ties_to_even_bare(x: Interval) -> BareResult {
    todo!()
}

fn round_ties_to_away_bare(x: Interval) -> BareResult {
    todo!()
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
    // Required Level 2 behavior:
    //
    // - Empty for x=Empty and bounded y;
    // - hull([inf(x)-inf(y), sup(x)-sup(y)]) for the valid bounded case;
    // - Entire for every Level 1 no-value case.
    todo!()
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
    rounding::midpoint(x.inf_raw(), x.sup_raw())
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

fn rad_from_mid_bare(x: Interval, midpoint: f64) -> f64 {
    if x.is_empty_raw() {
        return f64::NAN;
    }
    rounding::radius(x.inf_raw(), x.sup_raw(), midpoint)
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

fn disjoint_bare(x: Interval, y: Interval) -> bool {
    x.is_empty_raw() || y.is_empty_raw() || x.sup_raw() < y.inf_raw() || y.sup_raw() < x.inf_raw()
}

fn strict_or_same_infinity(left: f64, right: f64) -> bool {
    left < right || (left == right && left.is_infinite())
}

// Decoration helpers.

fn empty_result() -> BareResult {
    BareResult {
        interval: Interval::EMPTY,
        local: Decoration::Trv,
    }
}

fn continuous_unary_decoration(input: Interval, result: Interval) -> Decoration {
    if input.is_empty_raw() {
        Decoration::Trv
    } else if input.is_bounded_raw() && result.is_bounded_raw() {
        Decoration::Com
    } else {
        Decoration::Dac
    }
}

fn continuous_binary_decoration(x: Interval, y: Interval, result: Interval) -> Decoration {
    if x.is_empty_raw() || y.is_empty_raw() {
        Decoration::Trv
    } else if x.is_bounded_raw() && y.is_bounded_raw() && result.is_bounded_raw() {
        Decoration::Com
    } else {
        Decoration::Dac
    }
}

fn continuous_ternary_decoration(
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
