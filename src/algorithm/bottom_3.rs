use core::panic;

use super::utils::{
    error::{cross_entropy_error,sum_squares_error},
    gradient_descent::{numerical_gradient,gradient_descent},
    mnist::load_mnist,
    random::{ random_choice},
};
use ndarray::prelude::*;
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
    let y = arr1(&[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]).into_dyn(); //소프트 맥스 함수의 출력
    let t = arr1(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).into_dyn();
    //0일 확률 0.1,1일 확률 0.05, 1은 정답 레이블의 위치를 가르치는 원소 1 그외에는 0으로 표 기 => 원핫인코딩
    // // y= 2일 확률이 제일 높다고 판단함
    println!("오차제곱합:{}", sum_squares_error(&y, &t));
    // //0.09750000000000003
    let mut y = arr1(&[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]).into_dyn(); //소프트 맥스 함수의 출력
    println!("오차제곱합:{}", sum_squares_error(&y, &t));
    // //0.5974999999999999 =? 7이 가장 높다고 추정함=>
    /*교차 엔트로피 오차 */
    println!("교차 엔트로피:{}", cross_entropy_error(&y, &t));
    let y = arr1(&[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]).into_dyn(); //소프트 맥스 함수의 출력
    println!("교차 엔트로피:{}", cross_entropy_error(&y,  & t)); //=>오차제곱합 와 일치

    let x_train = load_mnist().x_train;
    let y_train = load_mnist().y_train;
    let x_test = load_mnist().x_test;
    let y_test = load_mnist().y_test;
    println!("X train Shape{:?}", x_train.clone().shape()); //60000,784
    println!("Y train Shape{:?}", y_train.clone().shape()); //60000 10

    let train_size = x_train.clone().shape()[0]; //trian_size
    let batch_size = 10;
    let batch_mask = random_choice(train_size, batch_size); //무작위로 원하는 개수만 꺼내기 =>무작위로 10개씩
    println!("무작위{:?}", batch_mask);
    let x_batch: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> = x_train.select(Axis(1), &batch_mask);
    println!("x_batch:{}", x_batch);
    // let y_batch: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>> =
    //     y_train.select(Axis(0), &batch_mask);
    // println!("y_train::{}", y_batch);
    // println!("{:?}", random_choice(60000, 10));
    // println!(
    //     "cross:{:?}",
    //     cross_entropy_error(&y_train.into_dyn(), &x_train.into_dyn())
    // );

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
    println!("미분:{}", numerical_diff(function_1, 10.0)); //0

  /*변수가 여러개=>편미분 */
    println!("편미분:{}", numerical_diff(function_tmp1, 5.0)); //0
    println!("편미분:{}", numerical_diff(function_tmp1, 3.0));
    println!("편미분:{}", numerical_diff(function_tmp2, 4.0));
}

fn function_1(x: f64) -> f64 {
    0.01 * x.powf(2.0) + 0.1 * x
}
fn function_2(x: &ArrayD<f64>) -> f64 {
    let rank = x.ndim();
    match rank {
        1 => {
            let x = x.clone().into_dimensionality::<Ix1>().unwrap();
            x[0].powf(2.0) + x[1].powf(2.0)
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
