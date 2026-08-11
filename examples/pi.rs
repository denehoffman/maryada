//! An example which estimates the constant π
use maryada::{Interval, IntervalOps};

/// The Basel problem says that π can be found from an infinite sum:
///
/// ```text
///        ∞
/// π²/6 = Σ 1/n².
///       n=1
/// ```
///
/// We can split the sum into two parts, a finite sum up to some `N` and the remaining infinite
/// term:
///
/// ```text
///        N        ∞
/// π²/6 = Σ 1/n² + Σ 1/n².
///       n=1     n=N+1
/// ```
///
/// Since
///
/// ```text
/// 1/(n(n+1)) < 1/n² < 1/(n(n-1)),
/// ```
///
/// ```text
///   ∞              ∞
///   Σ 1/(n(n+1)) = Σ (1/n - 1/(n+1)) = 1/(N+1),
/// n=N+1          n=N+1
/// ```
///
/// and
///
/// ```text
///   ∞              ∞
///   Σ 1/(n(n-1)) = Σ (1/(n-1) - 1/n)) = 1/N
/// n=N+1          n=N+1
/// ```
///
/// then
///
/// ```text
///           ∞
/// 1/(N+1) < Σ 1/n² < 1/N.
///         n=N+1
/// ```
///
/// We can then calculate the sum up to some term and then determine the distance from that value
/// to the upper bound obtained by propagating this inequality back through the original equation.
///
pub fn main() {
    #[allow(clippy::arithmetic_side_effects, clippy::expect_used)]
    fn backward_sum(n: usize) -> Interval {
        let mut s_n: Interval = 0.0.into();
        for i in (1..=n).rev() {
            let i = f64::from(u32::try_from(i).expect("example index fits in u32"));
            s_n = s_n + (1.0 / Interval::from(i).powi(2));
        }
        let n = f64::from(u32::try_from(n).expect("example bound fits in u32"));
        let t_n = 1.0 / Interval::new(n, n + 1.0);
        let s = s_n + t_n;
        (6.0 * s).sqrt()
    }

    let pi_interval = backward_sum(1_000_000);
    println!("{:?}", pi_interval.mid_rad());
}
