use std::ops::Index;

use super::weight;
use ndarray::prelude::*;
use polars::prelude::*;

struct Mnist {
    x_train: Array2<i32>,
    y_train: Array1<i32>,
}
impl Mnist {
    fn load_mnist() -> Mnist {
        let x_train = CsvReader::from_path("./dataset/mnist/x_train.csv")
            .unwrap()
            .has_header(false)
            .finish()
            .unwrap()
            .to_ndarray::<Int32Type>(IndexOrder::Fortran)
            .unwrap();
        let y_train = CsvReader::from_path("./dataset/mnist/y_train.csv")
            .unwrap()
            .has_header(false)
            .finish()
            .unwrap()
            .to_ndarray::<Int32Type>(IndexOrder::Fortran)
            .unwrap()
            .index_axis(Axis(1), 0)
            .to_owned();
        Mnist { x_train, y_train }
    }
}
/*신경망 학습 */
pub fn main() {
    // 오차제곱합
    let t = arr1(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let y = arr1(&[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]);

    println!("{}", sum_squares_error(&y, &t));
    println!("{}", cross_entropy_error(&y, &t));
    let x_train = Mnist::load_mnist().x_train;
    let y_train = Mnist::load_mnist().y_train;
    println!("{:?}", x_train.shape());
    println!("{:?}", y_train.shape());
}

fn sum_squares_error(y: &Array1<f64>, t: &Array1<f64>) -> f64 {
    let squared_diff = y
        .iter()
        .zip(t.iter())
        .map(|(&y_i, &t_i)| (y_i - t_i).powi(2))
        .sum::<f64>();
    0.5 * squared_diff
}

fn cross_entropy_error(y: &Array1<f64>, t: &Array1<f64>) -> f64 {
    let delta = 1e-7;
    let y = y.map(|&y_i| y_i + delta);

    let log_y = y.iter().zip(t.iter()).map(|(&y_i, &t_i)| t_i * y_i.ln());

    let neg_sum_log_y: f64 = log_y.map(|x| -x).sum();

    neg_sum_log_y
}
