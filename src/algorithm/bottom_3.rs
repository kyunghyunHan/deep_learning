use ndarray::prelude::*;

/*신경망 학습 */
pub fn main() {
    // 오차제곱합
    let t = arr1(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let y = arr1(&[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]);

    println!("{}", sum_squares_error(&y, &t));
}

fn sum_squares_error(y: &Array1<f64>, t: &Array1<f64>) -> f64 {
    let squared_diff = y
        .iter()
        .zip(t.iter())
        .map(|(&y_i, &t_i)| (y_i - t_i).powi(2))
        .sum::<f64>();
    0.5 * squared_diff
}
