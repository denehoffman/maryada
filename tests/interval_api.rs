use maryada::{DecoratedInterval, Interval, convex_hull, disjoint, interior, intersection, subset};

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
