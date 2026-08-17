//! Tests for the sealed real-interval convenience trait.

#![allow(clippy::arithmetic_side_effects)]

use core::fmt::Debug;

use maryada::{
    DecoratedInterval, Decoration, EnclosureOps, Interval, IntervalOps, add, decoration_part,
    set_dec, sqr,
};

fn exercise_common_api<T>()
where
    T: IntervalOps + Debug + PartialEq,
{
    let x = T::new(1.0, 2.0);
    let y = T::new(3.0, 4.0);

    assert_eq!(T::ZERO.bounds(), (0.0, 0.0));
    assert_eq!(T::ONE.bounds(), (1.0, 1.0));
    assert!(T::EMPTY.is_empty());
    assert!(T::ENTIRE.is_entire());
    assert_eq!(T::singleton(2.0).bounds(), (2.0, 2.0));

    assert_eq!((x + y).bounds(), (4.0, 6.0));
    assert_eq!((y - x).bounds(), (1.0, 3.0));
    assert_eq!((x * y).bounds(), (3.0, 8.0));
    assert_eq!((y / x).bounds(), (1.5, 4.0));
    assert_eq!((-x).bounds(), (-2.0, -1.0));

    let mut assigned = T::from(8.0);
    assigned += T::from(2.0);
    assert_eq!(assigned.bounds(), (10.0, 10.0));
    assigned -= T::from(2.0);
    assert_eq!(assigned.bounds(), (8.0, 8.0));
    assigned *= T::from(2.0);
    assert_eq!(assigned.bounds(), (16.0, 16.0));
    assigned /= T::from(2.0);
    assert_eq!(assigned.bounds(), (8.0, 8.0));
    assigned += 2.0;
    assigned -= 2.0;
    assigned *= 2.0;
    assigned /= 2.0;
    assert_eq!(assigned.bounds(), (8.0, 8.0));

    assert_eq!(x.sqr(), sqr(x));
    assert_eq!(x.powi(2), x.sqr());
    assert_eq!(x.mul_add(y, T::ONE), x * y + T::ONE);
    assert_eq!(x.hypot(T::ZERO), x.abs());
    assert_eq!(x.inner_rad(), 0.5);
    assert!(x.equal(T::new(1.0, 2.0)));
    assert!(x.subset(x.convex_hull(y)));
    assert_eq!(x.hull_value(4.0).bounds(), (1.0, 4.0));
    assert!(x.contains(1.5));
    assert!(T::ONE.is_singleton());
    assert!(x.is_bounded());
    assert!(x.intersects(y.convex_hull(x)));

    let (left, right) = x.bisect();
    assert_eq!(left.bounds(), (1.0, 1.5));
    assert_eq!(right.bounds(), (1.5, 2.0));
}

#[test]
fn common_api_is_generic_over_bare_and_decorated_intervals() {
    exercise_common_api::<Interval>();
    exercise_common_api::<DecoratedInterval>();
}

#[test]
fn methods_match_standard_free_functions() {
    let x = Interval::new(1.0, 2.0);
    let y = Interval::new(3.0, 4.0);

    assert_eq!(x + y, add(x, y));
    assert_eq!(x.sqr(), sqr(x));
    assert_eq!(x.mid_rad(), maryada::mid_rad(x));
    assert_eq!(x.cancel_plus(y), maryada::cancel_plus(x, y));
}

#[test]
fn decorated_methods_preserve_nai_and_decorations() {
    let nai = DecoratedInterval::NAI;
    assert!(nai.is_nai());
    assert!(nai.sqr().is_nai());
    assert!(!nai.intersects(DecoratedInterval::ENTIRE));
    let (left, right) = nai.bisect();
    assert!(left.is_nai());
    assert!(right.is_nai());

    let value = set_dec(Interval::new(1.0, 3.0), Decoration::Def);
    assert_eq!(decoration_part(value.sqr()), Decoration::Def);
    let (left, right) = value.bisect();
    assert_eq!(decoration_part(left), Decoration::Def);
    assert_eq!(decoration_part(right), Decoration::Def);
}
