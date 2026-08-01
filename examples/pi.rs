use maryada::Interval;

pub fn main() {
    fn backward_sum(n: usize) -> Interval {
        let mut s_n: Interval = 0.0.into();
        for i in (1..=n).rev() {
            s_n = s_n + (1.0 / Interval::from(i as f64).powi(2));
        }
        let t_n = 1.0 / Interval::new(n as f64, (n + 1) as f64);
        let s = s_n + t_n;
        (6.0 * s).sqrt()
    }

    let pi_interval = backward_sum(1_000_000);
    println!("{:?}", pi_interval.mid_rad());
}
