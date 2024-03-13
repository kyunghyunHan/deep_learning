use super::weight;
use ndarray::prelude::*;
use polars::prelude::*;
use ndarray_stats::QuantileExt;
struct Mnist {
    x_train: Array2<f64>,
    y_train: Array1<i64>,
    x_test: Array2<f64>,
    y_test: Array1<i64>,
}

impl Mnist {
    fn new() -> Self {
        let train_df = CsvReader::from_path("./datasets/mnist/train.csv")
            .unwrap()
            .finish()
            .unwrap();
        let test_df = CsvReader::from_path("./datasets/mnist/test.csv")
            .unwrap()
            .finish()
            .unwrap();
        let submission_df = CsvReader::from_path("./datasets/mnist/sample_submission.csv")
            .unwrap()
            .finish()
            .unwrap();

        let y_train = arr1(
            &train_df
                .column("label")
                .unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<i64>>(),
        );

        let y_test = arr1(
            &submission_df
                .column("Label")
                .unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<i64>>(),
        );

        let x_train = train_df
            .drop("label")
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();
        let x_test = test_df
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();
        Mnist {
            x_train,
            y_train,
            x_test,
            y_test,
        }
    }
}

#[derive(Clone)]
struct Network {
    w1: Array2<f64>,
    w2: Array2<f64>,
    w3: Array2<f64>,
    b1: Array1<f64>,
    b2: Array1<f64>,
    b3: Array1<f64>,
}
impl Network {
    fn init_network() -> Self {
        let w1_array: Vec<f64> = weight::w1::w1
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .collect();
        let w1 = ArrayView2::from_shape((784, 50), &w1_array)
            .unwrap()
            .to_owned();

        let w2_array: Vec<f64> = weight::w2::w2
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .collect();
        let w2 = ArrayView2::from_shape((50, 100), &w2_array)
            .unwrap()
            .to_owned();

        let w3_array: Vec<f64> = weight::w3::w3
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .collect();
        let w3 = ArrayView2::from_shape((100, 10), &w3_array)
            .unwrap()
            .to_owned();

        let b1 = arr1(&weight::b1::b1);
        let b2 = arr1(&weight::b2::b2);
        let b3 = arr1(&weight::b3::b3);
        Network {
            w1: w1,
            w2: w2,
            w3: w3,
            b1: b1,
            b2: b2,
            b3: b3,
        }
    }
    fn forward(self, x: ArrayD<f64>) -> ArrayD<f64> {
        let rank = x.ndim();
        let y: ArrayD<f64>;
        if rank == 1 {
            let x = x.into_dimensionality::<Ix1>().unwrap();

            let network = Network::init_network();
            let (w1, w2, w3) = (network.w1, network.w2, network.w3);
            let (b1, b2, b3) = (network.b1, network.b2, network.b3);
            println!("{:?}", w1.shape());
            println!("{:?}", w2.shape());
            println!("{:?}", w3.shape());
            println!("{:?}", x.shape());

            let a1 = x.dot(&w1) + b1;
            let z1 = sigmoid_function(&a1.into_dyn());
            let a2 = z1.into_dimensionality::<Ix1>().unwrap().dot(&w2) + b2;
            let z2 = sigmoid_function(&a2.into_dyn());
            let a3 = z2.into_dimensionality::<Ix1>().unwrap().dot(&w3) + b3;
            y = identity_function(&a3.into_dyn());
        } else if rank == 2 {
            let x = x.into_dimensionality::<Ix2>().unwrap();

            let network = Network::init_network();
            let (w1, w2, w3) = (network.w1, network.w2, network.w3);
            let (b1, b2, b3) = (network.b1, network.b2, network.b3);

            let a1 = x.dot(&w1) + b1;
            let z1 = sigmoid_function(&a1.into_dyn());
            let a2 = z1.into_dimensionality::<Ix2>().unwrap().dot(&w2) + b2;
            let z2 = sigmoid_function(&a2.into_dyn());
            let a3 = z2.into_dimensionality::<Ix2>().unwrap().dot(&w3) + b3;
            y = identity_function(&a3.into_dyn());
        } else {
            panic!("Unsupported rank: {}", rank);
        }
        y
    }
    fn predict(self, x: ArrayD<f64>) -> ArrayD<f64> {
        let (w1, w2, w3) = (self.w1, self.w2, self.w3);
        let (b1, b2, b3) = (&self.b1, &self.b2, &self.b3);
        let y: ArrayD<f64>;
        let rank = x.ndim();
        if rank == 1 {
            let x = x.into_dimensionality::<Ix1>().unwrap();
            let a1 = x.dot(&w1) + b1;
            let z1 = sigmoid_function(&a1.into_dyn());
            let a2: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>> = z1.into_dimensionality::<Ix1>().unwrap().dot(&w2) + b2;
            let z2 = sigmoid_function(&a2.into_dyn());
            let a3 = z2.into_dimensionality::<Ix1>().unwrap().dot(&w3) + b3;
            y = softmax(&a3.into_dyn())
        } else if rank == 2 {
            let x = x.into_dimensionality::<Ix2>().unwrap();
            let a1 = x.dot(&w1) + b1;
            let z1 = sigmoid_function(&a1.into_dyn());
            let a2 = z1.into_dimensionality::<Ix2>().unwrap().dot(&w2) + b2;
            let z2 = sigmoid_function(&a2.into_dyn());
            let a3 = z2.into_dimensionality::<Ix2>().unwrap().dot(&w3) + b3;
            y = softmax(&a3.into_dyn());
        } else {
            panic!("Unsupported rank: {}", rank);
        }

        y
    }
}
pub fn main() {
    let mnist = Mnist::new();
    let x_train = mnist.x_train;
    let y_train = mnist.y_train;
    let x_test = mnist.x_test;
    let y_test = mnist.y_test;
    let network = Network::init_network();
    println!("x_trian:{:?}", x_train.shape());
    println!("y_train:{:?}", y_train.shape());
    println!("x_test:{:?}", x_test.shape());
    println!("y_test:{:?}", y_test.shape());
    println!("{}",x_train.len());
    let a = x_train.slice(s![1..4, ..]); // 행 1에서 1까지, 모든 열
    println!("{:?}",a.shape());
    let mut accuracy_cnt =0f64;
    for i in 0..x_train.shape()[0]{
          println!("{}",accuracy_cnt);
          let y=network.clone().predict(x_train.row(i).into_owned().into_dyn());
          let p = y.into_dimensionality::<Ix1>().unwrap().argmax().unwrap();
          if p==*y_train.get(i).unwrap() as usize{
            accuracy_cnt+=1.0;
          }
          
    }
    println!("{}",accuracy_cnt/x_train.shape()[0] as f64)
}
fn sigmoid_function(x: &ArrayD<f64>) -> ArrayD<f64> {
    x.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}
fn softmax(a: &ArrayD<f64>) -> ArrayD<f64> {
    let c: f64 = a[a.argmax().unwrap()];
    let exp_a = a.mapv(|x| (x - c).exp());
    let sum_exp_a = exp_a.sum();
    exp_a / sum_exp_a
}
fn identity_function(x: &ArrayD<f64>) -> ArrayD<f64> {
    return x.clone();
}