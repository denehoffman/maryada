#![cfg(feature = "complex")]

use maryada::{ComplexBox, DecoratedInterval, Interval};

#[test]
fn complex_boxes_work_without_num_complex() {
    let x = ComplexBox::<Interval>::new(Interval::new(1.0, 2.0), Interval::new(3.0, 4.0));
    let y = ComplexBox::<Interval>::new(Interval::from(2.0), Interval::from(1.0));

    let sum = x + y;
    assert_eq!(sum.re.bounds(), (3.0, 4.0));
    assert_eq!(sum.im.bounds(), (4.0, 5.0));
    assert!(x.is_bounded());
    assert!(x.diameter().is_finite());
}

#[test]
fn complex_operators_cover_real_and_box_combinations() {
    type BareBox = ComplexBox<Interval>;
    type DecoratedBox = ComplexBox<DecoratedInterval>;

    fn bare(_: BareBox) {}
    fn decorated(_: DecoratedBox) {}

    macro_rules! check_op {
        ($op:tt) => {{
            let bare_box = BareBox::from(2.0);
            let decorated_box = DecoratedBox::from(3.0);
            let bare_interval = Interval::from(4.0);
            let decorated_interval = DecoratedInterval::from(5.0);
            let scalar = 6.0;

            bare(bare_box $op bare_box);
            decorated(bare_box $op decorated_box);
            decorated(decorated_box $op bare_box);
            decorated(decorated_box $op decorated_box);

            bare(bare_box $op bare_interval);
            bare(bare_interval $op bare_box);
            decorated(bare_box $op decorated_interval);
            decorated(decorated_interval $op bare_box);
            decorated(decorated_box $op bare_interval);
            decorated(bare_interval $op decorated_box);
            decorated(decorated_box $op decorated_interval);
            decorated(decorated_interval $op decorated_box);

            bare(bare_box $op scalar);
            bare(scalar $op bare_box);
            decorated(decorated_box $op scalar);
            decorated(scalar $op decorated_box);
        }};
    }

    check_op!(+);
    check_op!(-);
    check_op!(*);
    check_op!(/);
}
