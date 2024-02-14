use core::panic;

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use polars::prelude::*;
use rand::prelude::*;

/*신경망 학습 */
pub fn main() {
    /*
    학습이란 훈련데이터로부터 가중치 매개변수의 최적값을 자동으로 획득하는것
     */

    /*손실함수
    - 일반적으로  오차제곱합과 교차 엔트로피 오차 사용
     */
    //오차제곱합
    let y = arr1(&[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]); //소프트 맥스 함수의 출력
    let t = arr1(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    //0일 확률 0.1,1일 확률 0.05, 1은 정답 레이블의 위치를 가르치는 원소 1 그외에는 0으로 표 기 => 원핫인코딩
    // y= 2일 확률이 제일 높다고 판단함
    println!("오차제곱합:{}", sum_squares_error(&y, &t));
    //0.09750000000000003
    let y = arr1(&[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]); //소프트 맥스 함수의 출력
    println!("오차제곱합:{}", sum_squares_error(&y, &t));
    //0.5974999999999999 =? 7이 가장 높다고 추정함=>
    /*교차 엔트로피 오차 */
    println!(
        "교차 엔트로피:{}",
        cross_entropy_error(&y.into_dyn(), &t.clone().into_dyn())
    );
    let y = arr1(&[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]); //소프트 맥스 함수의 출력
    println!(
        "교차 엔트로피:{}",
        cross_entropy_error(&y.into_dyn(), &t.clone().into_dyn())
    ); //=>오차제곱합 와 일치

    let x_train = Mnist::load_mnist().x_train;
    let y_train = Mnist::load_mnist().y_train;
    println!("X train Shape{:?}", x_train.shape()); //60000,784
    println!("Y train Shape{:?}", y_train.shape()); //60000 10

    let train_size = x_train.shape()[0]; //trian_size
    let batch_size = 10;
    let batch_mask = random_choice(train_size, batch_size); //무작위로 원하는 개수만 꺼내기 =>무작위로 10개씩
    println!("무작위{:?}", batch_mask);
    let x_batch = x_train.select(Axis(0), &batch_mask);
    println!("x_batch:{}", x_batch);
    let y_batch: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> =
        y_train.select(Axis(0), &batch_mask);
    println!("y_train::{}", y_batch);
    println!("{:?}", random_choice(60000, 10));
    println!(
        "cross:{:?}",
        cross_entropy_error(&y_train.into_dyn(), &x_train.into_dyn())
    );

    /*미분
    마라톤 선수 10분에 2km
    속도는 0.2 1분에 0.2
    미분=>변화량

    10e4


     */
    //수치미분
    let x1 = Array1::range(0.0, 20.0, 0.1);
    let y1 = x1.map(|&elem| function_1(elem));
    println!("{:?}", y1);
    println!("미분:{}", numerical_diff(function_1, 5.0)); //0
    println!("미분:{}", numerical_diff(function_1, 5.0)); //0

    /*변수가 여러개=>편미분 */
    println!("편미분:{}", numerical_diff(function_tmp1, 5.0)); //0
    println!("편미분:{}", numerical_diff(function_tmp1, 3.0));
    println!("편미분:{}", numerical_diff(function_tmp2, 4.0));
    /*gradient */
    println!(
        "{}",
        numerical_gradient(function_2, arr1(&[3.0, 0.0]).into_dyn())
    );

    let init_x = arr1(&[-3.0, 4.0]);
    println!(
        "gradient_descent:{}",
        gradient_descent(function_2, init_x.into_dyn(), 0.1, 100)
    );
    println!(
        "학습률이 너무 큰 예{}",
        gradient_descent(function_2, arr1(&[-3.0, 4.0]).into_dyn(), 10.0, 100)
    );
    println!(
        "학습률이 너무 작은 예{}",
        gradient_descent(function_2, arr1(&[-3.0, 4.0]).into_dyn(), 1e-10, 100)
    );

    /*신경망 에서의 기울기 */
    let simple = SimpleNet {
        w: arr2(&[[0.0]]).into_dyn(),
    };
    let net = SimpleNet::_init_(simple);
    println!("가중치 매개변수:{:?}", net);
    let x = arr1(&[0.6, 0.9]);
    let p = net.clone().predict(x.clone().into_dyn());
    println!("{:?}", p);
    println!(
        "최대값의 인덱스:{}",
        p.into_dimensionality::<Ix1>().unwrap().argmax().unwrap()
    );

    let t = arr1(&[0f64, 0f64, 1f64]);

    println!(
        "loss:{:?}",
        SimpleNet::loss(net.clone(), x.clone().into_dyn(), t.into_dyn())
    );
}

fn sum_squares_error(y: &Array1<f64>, t: &Array1<f64>) -> f64 {
    let squared_diff = y
        .iter()
        .zip(t.iter())
        .map(|(&y_i, &t_i)| (y_i - t_i).powi(2))
        .sum::<f64>();
    0.5 * squared_diff
}
/*미해결 */
fn cross_entropy_error(y: &ArrayD<f64>, t: &ArrayD<f64>) -> f64 {
    let rank = y.ndim();
    let delta = 1e-7;
    match rank {
        1 => {
            let t = t
                .clone()
                .into_dimensionality::<Ix1>()
                .unwrap()
                .into_shape((1, t.len()))
                .unwrap();
            let y = y
                .clone()
                .into_dimensionality::<Ix1>()
                .unwrap()
                .into_shape((1, y.len()))
                .unwrap();
            //더하기
            let y = y.mapv(|val| val + delta);

            let mut errors = Vec::new();
            for (y_row, t_row) in y.outer_iter().zip(t.outer_iter()) {
                let error = -y_row
                    .iter()
                    .zip(t_row.iter())
                    .map(|(&y_i, &t_i)| t_i * y_i.ln())
                    .sum::<f64>();
                errors.push(error);
            }

            let batch_size = y.shape()[0] as f64;

            let total_error: f64 = errors.iter().sum();

            total_error / batch_size
        }
        2 => {
            let t = t
                .clone()
                .into_dimensionality::<Ix2>()
                .unwrap()
                .into_shape((1, t.len()))
                .unwrap();
            let y = y
                .clone()
                .into_dimensionality::<Ix2>()
                .unwrap()
                .into_shape((1, y.len()))
                .unwrap();
            let delta = 1e-7;
            let neg_sum_log_y: f64 = y
                .outer_iter()
                .zip(t.outer_iter())
                .map(|(y_row, t_row)| {
                    -y_row
                        .iter()
                        .zip(t_row.iter())
                        .map(|(&y_i, &t_i)| t_i * (y_i + delta).ln())
                        .sum::<f64>()
                })
                .sum();

            neg_sum_log_y / y.shape()[0] as f64
        }
        _ => {
            panic!("rank error")
        }
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

fn function_1(x: f64) -> f64 {
    0.01 * x.powf(2.0) + 0.1 * x
}
fn function_2(x: ArrayD<f64>) -> f64 {
    let rank = x.ndim();
    match rank {
        1 => {
            let x = x.into_dimensionality::<Ix1>().unwrap();
            return x[0].powf(2.0) + x[1].powf(2.0);
        }
        _ => {
            panic!("erorr이지")
        }
    }
}

/*편미분 */
fn function_tmp1(x0: f64) -> f64 {
    x0 * x0 + 4f64.powf(2.0)
}
fn function_tmp2(x1: f64) -> f64 {
    3f64.powf(2.0) + x1 * x1
}
/*미분 */
fn numerical_diff<F>(f: F, x: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = 1e-4; //0.0001
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

use ndarray::{Array1, Array2, Axis};

fn numerical_gradient<F>(f: F, x: ArrayD<f64>) -> ArrayD<f64>
where
    F: Fn(ArrayD<f64>) -> f64,
{
    let rank = x.ndim(); //rank설정
    let h = 1e-4;

    match rank {
        1 => {
            let mut x = x.clone().into_dimensionality::<Ix1>().unwrap();
            let mut grad: Array1<f64> = Array::zeros(x.raw_dim()); //x와 형상이 같은 배열을 생성
            for idx in 0..x.len() {
                let tmp_val = x[idx];
                //f(x+h)계산
                x[idx] = tmp_val + h;
                let fxh1 = f(x.clone().into_dyn());

                //f(x-h)계산
                x[idx] = tmp_val - h;
                let fxh2 = f(x.clone().into_dyn());

                grad[idx] = (fxh1 - fxh2) / (2.0 * h);
                x[idx] = tmp_val;
            }

            return grad.into_dyn();
        }
        2 => {
            let x = x.clone().into_dimensionality::<Ix2>().unwrap();
            let mut grad = Array2::zeros(x.raw_dim());
            for (idx, mut row) in x.clone().axis_iter_mut(Axis(0)).enumerate() {
                for (j, val) in row.iter_mut().enumerate() {
                    let tmp_val = *val;
                    *val = tmp_val + h;
                    let fxh1 = f(x.clone().into_dyn());
                    *val = tmp_val - h;
                    let fxh2 = f(x.clone().into_dyn());
                    grad[[idx, j]] = (fxh1 - fxh2) / (2.0 * h);
                    *val = tmp_val;
                }
            }
            return grad.into_dyn();
        }
        _ => {
            panic!("error")
        }
    }
}
/*경사하강법 */
fn gradient_descent<F>(f: F, init_x: ArrayD<f64>, ir: f64, step_num: i32) -> ArrayD<f64>
where
    F: Fn(ArrayD<f64>) -> f64,
{
    let mut x = init_x;
    for _ in 0..step_num {
        let grad = numerical_gradient(&f, x.clone().into_dyn());
        x = x - ir * grad;
    }
    x.into_dyn()
}
#[derive(Debug, Clone)]
struct SimpleNet {
    w: ArrayD<f64>,
}

impl SimpleNet {
    pub fn _init_(mut self) -> Self {
        let mut rng = rand::thread_rng();
        let matrix: [[f64; 3]; 2] = {
            let mut arr = [[0.0; 3]; 2];
            for i in 0..2 {
                for j in 0..3 {
                    arr[i][j] = rng.gen::<f64>();
                }
            }
            arr
        };
        self.w = arr2(&matrix).into_dyn();
        self
    }
    fn predict(self, x: ArrayD<f64>) -> ArrayD<f64> {
        let rank = x.ndim();

        match rank {
            1 => {
                let x = x.into_dimensionality::<Ix1>().unwrap();
                x.dot(&self.w.into_dimensionality::<Ix2>().unwrap())
                    .into_dyn()
            }
            2 => {
                let x = x.into_dimensionality::<Ix2>().unwrap();
                x.dot(&self.w.into_dimensionality::<Ix2>().unwrap())
                    .into_dyn()
            }
            _ => {
                panic!("not rank")
            }
        }
    }

    fn loss(self, x: ArrayD<f64>, t: ArrayD<f64>) -> f64 {
        let rank = x.ndim();
        match rank {
            1 => {
                let z = self.predict(x.clone());
                let y = softmax(z);
                let loss = cross_entropy_error(&y.into_dyn(), &t.into_dyn());
                loss
            }
            2 => {
                let z = self.predict(x.clone());
                let y = softmax(z);
                let loss = cross_entropy_error(&y.into_dyn(), &t.into_dyn());
                loss
            }
            _ => {
                panic!("not rank")
            }
        }
    }
    fn f_functsion(w: ArrayD<f64>) -> f64 {
        let x: ArrayD<f64> = Default::default();
        let t: ArrayD<f64> = Default::default();
        let net = SimpleNet { w: w };
        SimpleNet::loss(net, x, t)
    }
}
fn softmax(a: ArrayD<f64>) -> ArrayD<f64> {
    let rank = a.ndim();
    println!("{}", rank);
    if rank == 1 {
        let a = a.clone().into_dimensionality::<Ix1>().unwrap();
        let c: f64 = a[a.argmax().unwrap()];
        let exp_a = a.mapv(|x| (x - c).exp());
        let sum_exp_a = exp_a.sum();
        (exp_a / sum_exp_a).into_dyn()
    } else if rank == 2 {
        let a = a.clone().into_dimensionality::<Ix2>().unwrap();
        let exp_a = a.mapv(f64::exp);
        let sum_exp_a = exp_a.sum_axis(Axis(1));
        exp_a / sum_exp_a.insert_axis(Axis(1)).into_dyn()
    } else {
        panic!("rank error")
    }
}

#[derive(Clone)]
struct TwoLayerNet {
    w1: Array2<f64>,
    w2: Array2<f64>,
    b1: Array1<f64>,
    b2: Array1<f64>,
}
impl TwoLayerNet {
    fn new(
        self,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        weight_init_std: f64,
    ) -> TwoLayerNet {
        let mut rng = rand::thread_rng();
        let mut matrix = Array2::<f64>::zeros((input_size, hidden_size));
        //가중치
        TwoLayerNet {
            w1: weight_init_std
                * fill_with_random(
                    &mut Array2::<f64>::zeros((input_size, hidden_size)),
                    &mut rng,
                ),
            w2: weight_init_std
                * fill_with_random(
                    &mut Array2::<f64>::zeros((hidden_size, output_size)),
                    &mut rng,
                ),

            b1: Array1::<f64>::zeros(hidden_size),

            b2: Array1::<f64>::zeros(hidden_size),
        }
    }
    fn predict(self, x: ArrayD<f64>) -> ArrayD<f64> {
        let (w1, w2) = (self.w1, self.w2);
        let (b1, b2) = (self.b1, self.b2);

        let rank = x.ndim();

        match rank {
            1 => {
                let x = x.into_dimensionality::<Ix1>().unwrap();
                let a1 = x.dot(&w1) + b1;
                let z1 = sigmoid(a1.into_dyn()).into_dimensionality::<Ix1>().unwrap();
                let a2 = z1.dot(&w2) + b2;
                softmax(a2.into_dyn())
            }
            2 => {
                let x = x.into_dimensionality::<Ix2>().unwrap();
                let a1 = x.dot(&w1) + b1;
                let z1 = sigmoid(a1.into_dyn()).into_dimensionality::<Ix2>().unwrap();
                let a2 = z1.dot(&w2) + b2;
                softmax(a2.into_dyn())
            }
            _ => {
                panic!("predict rank error");
            }
        }
    }

    pub fn loss(self, x: ArrayD<f64>, t: ArrayD<f64>) -> f64 {
        cross_entropy_error(&x.into_dyn(), &t.into_dyn())
    }
    fn accuracy(self, x: ArrayD<f64>, t: ArrayD<f64>) -> f64 {
        let rank = x.ndim();

        match rank {
            1 => {
                let x = x.into_dimensionality::<Ix1>().unwrap();
                let y = self
                    .predict(x.clone().into_dyn())
                    .into_dimensionality::<Ix1>()
                    .unwrap();
                let y = y.argmax().unwrap();
                let t = t.into_dimensionality::<Ix1>().unwrap().argmax().unwrap();
                let accuracy = if y == t {
                    y as f64 + t as f64 / x.shape()[0] as f64
                } else {
                    0f64
                };
                accuracy
            }
            2 => {
                let y = self.predict(x.clone().into_dyn());
                let y_argmax = y
                    .into_dimensionality::<Ix2>()
                    .unwrap()
                    .outer_iter()
                    .map(|row| row.argmax().unwrap())
                    .collect::<Vec<_>>();
                let t_argmax = t
                    .into_dimensionality::<Ix2>()
                    .unwrap()
                    .outer_iter()
                    .map(|row| row.argmax().unwrap())
                    .collect::<Vec<_>>();
                let num_equal = y_argmax
                    .iter()
                    .zip(t_argmax.iter())
                    .filter(|&(yi, ti)| yi == ti)
                    .count();
                let accuracy = num_equal as f64 / x.shape()[0] as f64;
                accuracy
            }
            _ => {
                panic!("accuracy rank error")
            }
        }
    }

    fn numerical_gradient(
        self,
        x: ArrayD<f64>,
        t: ArrayD<f64>,
    ) -> (ArrayD<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        let (w1, w2, b1, b2) = (
            numerical_gradient(
                |x| self.clone().loss(x.clone(), t.clone()),
                self.clone().w1.into_dyn(),
            ),
            arr2(&[[1.0]]),
            arr2(&[[1.0]]),
            arr2(&[[1.0]]),
        );
        (w1, w2, b1, b2)
    }
}
fn fill_with_random(matrix: &mut Array2<f64>, rng: &mut impl Rng) -> Array2<f64> {
    let mut view = matrix.view_mut();

    for mut row in view.genrows_mut() {
        for elem in row.iter_mut() {
            *elem = rng.gen::<f64>();
        }
    }
    view.to_owned()
}
//시그모이드
fn sigmoid(x: ArrayD<f64>) -> ArrayD<f64> {
    x.mapv(|element| 1.0 / (1.0 + (-element).exp()))
}

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
