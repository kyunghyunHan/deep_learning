use ndarray::prelude::*;

pub fn sum_squares_error(y: &ArrayD<f64>, t: &ArrayD<f64>) -> f64 {
    0.5 * y
        .iter()
        .zip(t.iter())
        .map(|(&y_i, &t_i)| (y_i - t_i).powi(2))
        .sum::<f64>()
}

pub fn cross_entropy_error(y: &ArrayD<f64>, t: &ArrayD<f64>) -> f64 {
    let delta = 1e-7;
    match y.ndim() {
        1 => {
            let batch_size = y.len() as f64;
            let loss = -t * (y + delta).mapv(f64::ln);
            loss.sum() / batch_size
        }
        2 => {
            let batch_size = y.shape()[0] as f64;
            let loss = -t * (y + delta).mapv(f64::ln);
            loss.sum() / batch_size
        }
        _ => panic!("rank error"),
    }
}

