use core::cmp::Ordering;

use maryada::{
    DecoratedInterval, Decoration, Interval, Signal, SignalFlags, convex_hull, decoration_part,
    disjoint, interior, intersection, set_dec, subset,
};

#[test]
fn constructors_and_set_operations_follow_the_set_based_model() {
    assert!(maryada::new::<Interval>(2.0, 1.0).is_empty());
    assert!(maryada::singleton::<Interval>(f64::NAN).is_empty());
    assert!(maryada::new::<DecoratedInterval>(2.0, 1.0).is_nai());
    assert!(maryada::singleton::<DecoratedInterval>(f64::NAN).is_nai());

    let inner = Interval::new(1.0, 2.0);
    let outer = Interval::new(0.0, 3.0);
    let apart = Interval::new(4.0, 5.0);

    assert!(subset(inner, outer));
    assert!(interior(inner, outer));
    assert!(disjoint(inner, apart));
    assert_eq!(intersection(inner, outer), inner);
    assert_eq!(convex_hull(inner, outer), outer);
    assert_eq!(intersection(inner, apart), Interval::EMPTY);
    assert_eq!(convex_hull(Interval::EMPTY, outer), outer);
}

#[test]
#[allow(clippy::op_ref)]
fn ergonomic_protocols_preserve_interval_semantics_and_edge_states() {
    let x = Interval::new(1.0, 2.0);
    let y = Interval::new(3.0, 4.0);

    assert_eq!(&x + &y, maryada::add(x, y));
    assert_eq!(x - y, maryada::sub(x, y));
    assert_eq!(&x * 2.0, maryada::mul(x, 2.0.into()));
    assert_eq!(2.0 * &x, maryada::mul(2.0.into(), x));
    assert_eq!(&x / 2.0, maryada::div(x, 2.0.into()));
    assert_eq!(2.0 / &x, maryada::div(2.0.into(), x));
    assert_eq!(-&x, maryada::neg(x));
    assert_eq!(x.powi(3), Interval::new(1.0, 8.0));

    let hypot = x.hypot(&y);
    assert!(hypot.contains(10.0_f64.sqrt()));
    assert!(hypot.contains(20.0_f64.sqrt()));
    assert_eq!(x.bounds(), (1.0, 2.0));
    let (midpoint, radius) = x.mid_rad();
    assert!(midpoint - radius <= x.inf());
    assert!(midpoint + radius >= x.sup());

    let scalar = 1.5;
    assert_eq!(Interval::from(&scalar), Interval::from(scalar));
    assert_eq!(
        DecoratedInterval::from(&scalar),
        DecoratedInterval::from(scalar)
    );
    assert_eq!(DecoratedInterval::from(&x), DecoratedInterval::from(x));
    assert_eq!(Interval::from(&DecoratedInterval::from(x)), x);
    assert_eq!(
        Interval::from(DecoratedInterval::from(scalar)),
        scalar.into()
    );

    let parsed: Interval = "[1/3,2/3]".parse().unwrap();
    let displayed = parsed.to_string();
    assert_eq!(displayed.parse::<Interval>().unwrap(), parsed);
    assert!("not an interval".parse::<Interval>().is_err());

    let decorated: DecoratedInterval = "[1,2]_def".parse().unwrap();
    assert_eq!(decoration_part(decorated), Decoration::Def);
    assert_eq!(decorated.powi(3).bounds(), (1.0, 8.0));
    let decorated_y = set_dec(y, Decoration::Def);
    let decorated_hypot = decorated.hypot(&decorated_y);
    assert!(decorated_hypot.contains(10.0_f64.sqrt()));
    assert!(decorated_hypot.contains(20.0_f64.sqrt()));
    let (midpoint, radius) = decorated.mid_rad();
    assert!(midpoint - radius <= decorated.inf());
    assert!(midpoint + radius >= decorated.sup());
    assert_eq!(
        decorated.to_string().parse::<DecoratedInterval>().unwrap(),
        decorated
    );
    assert!("[empty]_com".parse::<DecoratedInterval>().is_err());

    let mut signals = SignalFlags::NONE;
    assert!(Interval::new(2.0, 1.0).decorate(&mut signals).is_nai());
    assert!(signals.contains(Signal::UndefinedOperation));
    signals.clear();
    assert!(Interval::new(f64::NAN, 1.0).decorate(&mut signals).is_nai());
    assert!(signals.contains(Signal::UndefinedOperation));

    assert_eq!(x.partial_cmp(&x), Some(Ordering::Equal));
    assert_eq!(
        x.partial_cmp(&Interval::new(0.0, 3.0)),
        Some(Ordering::Less)
    );
    assert_eq!(
        Interval::new(0.0, 3.0).partial_cmp(&x),
        Some(Ordering::Greater)
    );
    assert_eq!(x.partial_cmp(&Interval::new(1.5, 2.5)), None);

    let same_set_lower_decoration = set_dec(x, Decoration::Trv);
    let same_set_higher_decoration = set_dec(x, Decoration::Com);
    assert_eq!(
        same_set_lower_decoration.partial_cmp(&same_set_higher_decoration),
        None
    );
    assert_eq!(
        DecoratedInterval::NAI.partial_cmp(&DecoratedInterval::NAI),
        Some(Ordering::Equal)
    );

    let (left, right) = x.bisect();
    assert_eq!(left.convex_hull(&right), x);
    assert_eq!(Interval::EMPTY.bisect(), (Interval::EMPTY, Interval::EMPTY));
    let (left, right) = decorated.bisect();
    assert_eq!(decoration_part(left), Decoration::Def);
    assert_eq!(decoration_part(right), Decoration::Def);
    assert_eq!(
        DecoratedInterval::NAI.bisect(),
        (DecoratedInterval::NAI, DecoratedInterval::NAI)
    );

    assert_eq!(x.hull_value(5.0), Interval::new(1.0, 5.0));
    assert!(x.contains(1.5));
    assert!(!x.contains(f64::INFINITY));
    assert!(x.intersects(&Interval::new(2.0, 3.0)));
    assert!(x.is_bounded());
    assert!(!Interval::ENTIRE.is_bounded());
    assert!(Interval::from(1.0).is_singleton());
    assert!(decorated.is_bounded());
    assert_eq!(decorated.decoration(), Decoration::Def);
    assert_eq!(decorated.hull_value(5.0).bounds(), (1.0, 5.0));
    assert!(decorated.intersects(&DecoratedInterval::from(2.0)));
    assert!(!decorated.is_entire());
    assert!(DecoratedInterval::ENTIRE.is_entire());

    let decorated_outer = set_dec(Interval::new(0.0, 3.0), Decoration::Def);
    assert_eq!(
        decorated.partial_cmp(&decorated_outer),
        Some(Ordering::Less)
    );
    assert_eq!(
        decorated_outer.partial_cmp(&decorated),
        Some(Ordering::Greater)
    );
    let decorated_overlap = set_dec(Interval::new(1.5, 2.5), Decoration::Def);
    assert_eq!(decorated.partial_cmp(&decorated_overlap), None);

    let mut invalid_signals = SignalFlags::NONE;
    let invalid = Interval::from_be_bytes(&[0; 3], &mut invalid_signals);
    assert!(invalid.is_empty());
    assert!(invalid_signals.contains(Signal::InvalidOperand));
}
