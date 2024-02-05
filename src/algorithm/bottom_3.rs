use super::weight;
use ndarray::prelude::*;
use polars::prelude::*;
use rand::prelude::*;
use std::ops::Index;

struct Mnist {
    x_train: Array2<i32>,
    y_train: Array2<i32>,
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
            .unwrap();
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

    let train_size = x_train.shape()[0];
    let batch_size = 10;
    let batch_mask = random_choice(train_size, batch_size);
    let x_batch = x_train.select(Axis(0), &batch_mask);
    let y_batch: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>> =
        y_train.select(Axis(0), &batch_mask);

    println!("{:?}", random_choice(60000, 10));
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
fn cross_entropy_error_arr2(y: &Array2<f64>, t: &Array2<f64>) -> f64 {
    if y.ndim() == 1 {
        let t = t.clone().into_shape((1, t.len())).unwrap();
        let y = y.clone().into_shape((1, y.len())).unwrap();

        let batch_size = y.shape()[0];
        return -y.iter().zip(t.iter()).map(|(&y_i, &t_i)| t_i * y_i.ln()).sum::<f64>() / batch_size as f64;
    } else {
        let batch_size = y.shape()[0];
        return -y.iter().zip(t.iter()).map(|(&y_i, &t_i)| t_i * y_i.ln()).sum::<f64>() / batch_size as f64;
    }
}
fn random_choice(train_size: usize, batch_size: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let batch_mask: Vec<usize> = (0..batch_size)
        .map(|_| rng.gen_range(0..train_size))
        .collect();
    batch_mask
}


