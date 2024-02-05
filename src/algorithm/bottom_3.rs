use super::weight;
use ndarray::prelude::*;
use polars::prelude::*;
use rand::prelude::*;
use std::ops::Index;

struct Mnist {
    x_train: Array2<f64>,
    y_train: Array2<f64>,
}
impl Mnist {
    fn load_mnist() -> Mnist {
        let x_train = CsvReader::from_path("./dataset/mnist/x_train.csv")
            .unwrap()
            .has_header(false)
            .finish()
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();
        let y_train = CsvReader::from_path("./dataset/mnist/y_train.csv")
            .unwrap()
            .has_header(false)
            .finish()
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
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
    let y_batch: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> =
        y_train.select(Axis(0), &batch_mask);

    println!("{:?}", random_choice(60000, 10));
    println!("{:?}", cross_entropy_error_arr2(&y_train, &x_train));
    let x1 = Array1::range(0.0, 20.0, 0.1);
    let y1 = x1.map(|&elem| function_1(elem));
    println!("{:?}", y1);
    println!("{}",numerical_diff(function_tmp1,5.0));//0
    println!("{}",numerical_diff(function_tmp1,3.0));
    println!("{}",numerical_gradient(function_2,arr1(&[3.0,0.0])));



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
/* */
fn random_choice(train_size: usize, batch_size: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let batch_mask: Vec<usize> = (0..batch_size)
        .map(|_| rng.gen_range(0..train_size))
        .collect();
    batch_mask
}

fn function_1(x:f64)->f64{
    0.01 * x.powf(2.0) +0.1*x
}
fn function_2 (x:&[f64])->f64{
    x[0].powf(2.0) + x[1].powf(2.0)
}
/*편미분 */
fn function_tmp1(x0:f64)->f64{
    x0* x0 +4f64.powf(2.0)
}
/*미분 */
fn numerical_diff<F>(f:F,x:f64)->f64
where
    F:Fn(f64)->f64
{
   let h = 1e-4;//0.0001
   return (f(x+h)-f(x-h))/(2.0*h)
}

fn numerical_gradient<F>(f:F,x:Array1<f64>)->Array1<f64>
where
    F:Fn(Array1<f64>)->f64
{
    let mut x_clone = x.clone(); // Clone x before entering the loop
   let h= 1e-4;
   let mut grad:Array1<f64> = Array::zeros(x.raw_dim());
   for idx in 0..x.len(){
    let tmp_val = x_clone[idx];
    //f(x+h)계산
    x_clone[idx] = tmp_val+h;
    let fxh1= f(x_clone.clone());

    //f(x-h)계산
    x_clone[idx]= tmp_val-h;
    let fxh2= f(x_clone.clone());

    grad[idx]= (fxh1-fxh2)/(2.0 * h);
    x_clone[idx]= tmp_val;
   }

   grad
}