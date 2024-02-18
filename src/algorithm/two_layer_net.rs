use super::utils::{
    activation::*,
    error::{cross_entropy_error, sum_squares_error},
    gradient_descent::numerical_gradient,
    mnist::load_mnist,
    random::{fill_with_random, random_choice},
};
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use std::cell::RefCell;
use std::rc::Rc;
#[derive(Clone)]
struct TwoLayerNet {
    pub w1: ArrayD<f64>,
    pub w2: ArrayD<f64>,
    pub b1: ArrayD<f64>,
    pub b2: ArrayD<f64>,
}
impl TwoLayerNet {
    fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        weight_init_std: f64,
    ) -> TwoLayerNet {
        let mut rng = rand::thread_rng();
        //가중치

        TwoLayerNet {
            w1: weight_init_std
                * fill_with_random(
                    &mut Array2::<f64>::zeros((input_size, hidden_size)),
                    &mut rng,
                )
                .into_dyn(),
            w2: weight_init_std
                * fill_with_random(
                    &mut Array2::<f64>::zeros((hidden_size, output_size)),
                    &mut rng,
                )
                .into_dyn(),

            b1: Array2::<f64>::zeros((1,hidden_size)).into_dyn(),

            b2: Array2::<f64>::zeros((1,output_size)).into_dyn(),
        }
    }
    fn predict(&self, x: &ArrayD<f64>) -> ArrayD<f64> {
        let (w1, w2) = (self.clone().w1, self.clone().w2);
        let (b1, b2) = (self.clone().b1, self.clone().b2);
        match x.ndim() {
            1 => {
                let x = x.clone().into_dimensionality::<Ix1>().unwrap();
                let a1 = x.dot(&w1.into_dimensionality::<Ix2>().unwrap()) + b1;
                let z1 = sigmoid(a1.into_dyn()).into_dimensionality::<Ix1>().unwrap();
                let a2 = z1.dot(&w2.into_dimensionality::<Ix2>().unwrap()) + b2;
                softmax(a2.into_dyn())
            }
            2 => {
                let x = x.clone().into_dimensionality::<Ix2>().unwrap();
                let a1 = x.dot(&w1.into_dimensionality::<Ix2>().unwrap()) + b1;
                let z1 = sigmoid(a1.into_dyn()).into_dimensionality::<Ix2>().unwrap();
                let a2 = z1.dot(&w2.into_dimensionality::<Ix2>().unwrap()) + b2;
                softmax(a2.into_dyn())
            }
            _ => {
                panic!("predict rank error");
            }
        }
    }

    pub fn loss(&self, x: &ArrayD<f64>, t: &ArrayD<f64>) -> f64 {
        let y = self.predict(x);
        cross_entropy_error(&y.into_dyn(), &t)
    }
    fn accuracy(self, x: &ArrayD<f64>, t: &ArrayD<f64>) -> f64 {
        let rank = x.ndim();
        match rank {
            1 => {
                let y = self.predict(&x).into_dimensionality::<Ix1>().unwrap();
                let y = y.clone().argmax().unwrap();
                let t = t
                    .clone()
                    .into_dimensionality::<Ix1>()
                    .unwrap()
                    .argmax()
                    .unwrap();
                let accuracy = if y == t {
                    y as f64 + t as f64 / x.shape()[0] as f64
                } else {
                    0f64
                };
                accuracy
            }
            2 => {
                let y = self.predict(x);

                let y_argmax = y
                    .into_dimensionality::<Ix2>()
                    .unwrap()
                    .outer_iter()
                    .map(|row| row.argmax().unwrap())
                    .collect::<Vec<_>>();
                let t_argmax = t
                    .clone()
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
    /*가중치 매개변수의 기울기를 구하기 */
    fn numerical_gradient(
        &mut self,
        x: &ArrayD<f64>,
        t: &ArrayD<f64>,
    ) -> (ArrayD<f64>, ArrayD<f64>, ArrayD<f64>, ArrayD<f64>) {
        let p_net = self as *const TwoLayerNet;
        let f = || unsafe {
            (*p_net).loss(x, t)
        };
            let (w1, w2, b1, b2) = (
                numerical_gradient(&f, &mut self.w1),

                numerical_gradient(&f, &mut self.w2),

                numerical_gradient(&f, &mut self.b1),

                numerical_gradient(&f, &mut self.b2),

            );

            (w1, w2, b1, b2)
       
    }
    /* numerical_gradient의 성능 개선판*/
    fn gradient(self, x: ArrayD<f64>, t: ArrayD<f64>) {}
}

pub fn main() {
    let mnist = load_mnist();
    let x_train = mnist.x_train;
    let y_train = mnist.y_train;
    // let x_test = mnist.x_test;
    // let y_test =mnist.y_test;

    //하이퍼 파라미터
    let iter_num = 10000; //반복횟수
    let train_size = x_train.clone().shape()[0];
    let batch_size = 100; //미니배치 사이즈
    let learning_rate = 0.1;

    let mut network = TwoLayerNet::new(784, 50, 10, 0.01);
    //저장공간
    let mut train_loss_list: Vec<f64> = vec![];
    let mut train_acc_list: Vec<f64> = vec![];
    let mut test_acc_list: Vec<f64> = vec![];

    let iter_per_epoch = usize::max(train_size / batch_size, 1);
    //여기부터 문제
    for i in 0..iter_num {
        //미니배치 획득
        let batch_mask = random_choice(train_size, batch_size);
        let x_batch = x_train.clone().select(Axis(0), &batch_mask).into_dyn();
        let t_batch = y_train.clone().select(Axis(0), &batch_mask).into_dyn();

        //기울기 계산
        //여기부터 문제
        let (w1, w2, b1, b2) = network.numerical_gradient(&x_batch, &t_batch);
        network.w1 -= &(learning_rate * w1);
        network.w2 -= &(learning_rate * w2);
        network.b1 -= &(learning_rate * b1);
        network.b2 -= &(learning_rate * b2);
        //매개변수 갱신
        let loss = network.loss(&x_batch, &t_batch);
        train_loss_list.push(loss);
        //학습 경과기록
        //1 epoch당 정확도 계싼
        println!("에포크 시작");
        if i % iter_per_epoch == 0 {
            let train_acc = network
                .clone()
                .accuracy(&x_train.clone().into_dyn(), &y_train.clone().into_dyn());

            train_acc_list.push(train_acc);
            println!("{}", train_acc);
        }
        // println!("{:?}", train_loss_list);
    }
}
