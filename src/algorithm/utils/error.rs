use ndarray::prelude::*;

pub fn sum_squares_error(y: &Array1<f64>, t: &Array1<f64>) -> f64 {
    let squared_diff = y
        .iter()
        .zip(t.iter())
        .map(|(&y_i, &t_i)| (y_i - t_i).powi(2))
        .sum::<f64>();
    0.5 * squared_diff
}
/*해결 */
pub fn cross_entropy_error(y: &ArrayD<f64>, t: &ArrayD<f64>) -> f64 {
    let delta = 1e-7;
    match y.ndim() {
        1 => {
            let y = y
                .clone()
                .into_shape((1, y.len()))
                .unwrap();
            let t = t.clone().into_shape((1, t.len())).unwrap();
            let batch_size = y.shape()[0] as f64;
            let y = y
                .iter()
                .zip(t.iter())
                .map(|(&y, &t)| t * (y + delta).ln())
                .sum::<f64>();
            -y / batch_size
        }

        2 => {
            let t = t.clone().into_dimensionality::<Ix2>().unwrap();
            let y = y.clone().into_dimensionality::<Ix2>().unwrap();
            let batch_size = y.shape()[0] as f64;
            let y = y
                .iter()
                .zip(t.iter())
                .map(|(&y, &t)| t * (y + delta).ln())
                .sum::<f64>();
            y / batch_size
        }
        _ => {
            panic!("rank error")
        }
    }
}
