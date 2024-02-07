use super::weight;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use plotters::prelude::*;
use polars::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

/*신경망 
입력층 - 은닉충 - 출력충

x1과 x2라는 두신호를 입력받아 y를 출력하는 퍼셉트론


b=편향
w1,w2=가중치 

편향을 표시한다면?=>그림
가중치가 b이고 입력이 1인 뉴런이 추가, 이퍼셉트론의 동작은 각 신호들에 가중치를 곱한 후 다음뉴런에 전달 그합이 0이 넘으면 1을 출력,그렇지 않으면 0을 출력
h(x)함수는 입력이 0을 넘으면 1을 반환 않으면 0을 반환


h(x)=입력신호의 총합을 출력신호로 변환하는 함수를  일반적으로 활성화 함수 2단계로 처리

가중치 신호를 조합한 결과가 a라는 노드가 되고 활성화 함수 h ()를 통과하여 y라는 노드로 변환되는 과정 (뉴런==노드)

왼쪽은 일반적이 뉴런 오른쪽은 활성화 처리 과정을 명시한 뉴런

*/

pub fn main() {

    /*sigmoid
    exp(-x) e^-x를 뜻함 ,e는 자연상수 2.7182..의 값을 갖는 실수
    신경망에서는 활성화  함수로 시그모이드 함수를 이용하여 신호를변환 그 변환된 신호를 다음 뉴런에 전달
    
    
     */
    let x = arr1(&[-5.0, 5.0, 0.1]);
    let y = step_function(&x);
    /*계단함수 구현 */
    let x = Array::range(-5.0, 5., 0.1);
    let y = step_function(&x);

    // 그래프 그리기
    let root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> =
        BitMapBackend::new("step_function.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    //x축 -6.0 부터 6까지
    //y축 -0.1부터 1.1까지
    let mut chart = ChartBuilder::on(&root)
        .caption("step_function", ("sans-serif", 50))
        .build_cartesian_2d(-6.0..6.0, -0.1..1.1)
        .unwrap();

    chart.configure_mesh().draw().unwrap();
    chart
        .draw_series(LineSeries::new(
            x.iter().cloned().zip(y.iter().cloned()),
            &BLUE,
        ))
        .unwrap();

    /*시그모이드 함수 구현*/
    let x = Array1::from(vec![-1.0, 1.0, 2.0]);
    // println!("{}", sigmoid(&x));

    let t = arr1(&[1.0, 2.0, 3.0]);
    println!("{}", 1.0 + &t);

    println!("{}", 1.0 / &t);

    let x = Array::range(-5.0, 5., 0.1);
    // let y: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>> = sigmoid(&x);

    // 그래프 그리기
    let root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> =
        BitMapBackend::new("sigmoid_function.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    //x축 -6.0 부터 6까지
    //y축 -0.1부터 1.1까지
    let mut chart = ChartBuilder::on(&root)
        .caption("sigmoid", ("sans-serif", 50))
        .build_cartesian_2d(-6.0..6.0, -0.1..1.1)
        .unwrap();

    chart.configure_mesh().draw().unwrap();
    chart
        .draw_series(LineSeries::new(
            x.iter().cloned().zip(y.iter().cloned()),
            &BLUE,
        ))
        .unwrap();
    chart
        .draw_series(LineSeries::new(
            x.iter().cloned().zip(y.iter().cloned()),
            &BLUE,
        ))
        .unwrap();

    /*계단 함수와 비교 */

    let y2: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>> = step_function(&x);

    // 그래프 그리기
    let root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> =
        BitMapBackend::new("sigmoid_function.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    //x축 -6.0 부터 6까지
    //y축 -0.1부터 1.1까지
    let mut chart = ChartBuilder::on(&root)
        .caption("sigmoid", ("sans-serif", 50))
        .build_cartesian_2d(-6.0..6.0, -0.1..1.1)
        .unwrap();

    chart.configure_mesh().draw().unwrap();
    chart
        .draw_series(LineSeries::new(
            x.iter().cloned().zip(y.iter().cloned()),
            &BLUE,
        ))
        .unwrap();
    chart
        .draw_series(LineSeries::new(
            x.iter().cloned().zip(y2.iter().cloned()),
            &RED,
        ))
        .unwrap();

    /*다차원 배열 */
    let a = arr1(&[1, 2, 3, 4]);
    print!("{}", a);

    //차원의 수 확인
    a.ndim();
    //형상:원소 개로 구성
    a.shape();

    let b = arr2(&[[1, 2], [3, 4], [5, 6]]);
    println!("{}", b);
    b.ndim();
    b.shape();

    /*행렬의 곱*/
    let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>> = arr2(&[[1, 2], [3, 4]]);
    println!("{:?}", a.shape());

    let b = arr2(&[[5, 6], [7, 8]]);
    println!("{:?}", b.shape());

    println!("{}", a.dot(&b));

    let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>> = arr2(&[[1, 2, 3], [4, 5, 6]]);
    println!("{:?}", a.shape());

    let b: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>> = arr2(&[[1, 2], [3, 4], [5, 6]]);
    println!("{:?}", b.shape());

    println!("{}", a.dot(&b));

    let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>> = arr2(&[[1, 2, 3], [4, 5, 6]]);
    println!("{:?}", a.shape());

    let c = arr1(&[[1, 2], [3, 4]]);
    println!("{:?}", c.shape());

    let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>> = arr2(&[[1, 2], [3, 4], [5, 6]]);
    println!("{:?}", a.shape());

    let b = arr1(&[7, 8]);
    println!("{:?}", b.shape());
    println!("{}", a.dot(&b));

    /*신경망 행렬곱 */
    let x = arr1(&[1, 2]);
    println!("{:?}", x.shape());

    let w = arr2(&[[1, 3, 5], [2, 4, 6]]);
    println!("{:?}", w.shape());
    println!("{}", w);

    let y = x.dot(&w);
    println!("{}", y);
    /*신경망 구현 */
    let x = arr1(&[1.0, 0.5]);
    let w1 = arr2(&[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]);
    let b1: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>> = arr1(&[0.1, 0.2, 0.3]);

    println!("{:?}", w1.shape());
    println!("{:?}", x.shape());
    println!("{:?}", b1.shape());

    let a1 = x.dot(&w1) + b1;
    println!("{:?}", a1.shape());

    // let z1 = sigmoid(&a1);
    println!("{}", a1);
    // println!("{}", z1);

    let w2 = arr2(&[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]);
    let b2 = arr1(&[0.1, 0.2]);

    // println!("{:?}", z1.shape());
    println!("{:?}", w2.shape());
    println!("{:?}", b2.shape());

    // let a2 = z1.dot(&w2) + b2;
    // let z2 = sigmoid(&a2);

    let w3 = arr2(&[[0.1, 0.3], [0.2, 0.4]]);
    let b3 = arr1(&[0.1, 0.2]);

    // let a3 = z2.dot(&w3) + b3;

    let y = idenity_function(&x);

    /*항등함수와 소프트맥스 함수 구현 */
    let a = arr1(&[0.3, 2.9, 4.0]);
    let exp_a = &a.map(|val: &f64| f64::exp(*val));
    println!("{}", exp_a);
    let sum_exp_a = exp_a.sum();
    println!("{}", sum_exp_a);

    let y = exp_a / sum_exp_a;
    println!("{}", &y);

    //소프트맥스
    let a = arr1(&[1010.0, 1000.0, 990.0]);
    let exp_a = a.mapv(|x| f64::exp(x));
    let exp_a_sum = exp_a.sum();
    println!("{}", exp_a / exp_a_sum); //Nan,Nan,Nan

    let c: f64 = a[a.argmax().unwrap()];
    println!("{}", &a - c);

    println!(
        "{}",
        (&a - c).mapv(|x| f64::exp(x)) / (&a - c).mapv(|x| f64::exp(x)).sum()
    );

    let a = arr1(&[0.3, 2.9, 4.0]);
    // let y = softmax(&a);
    println!("{}", y);
    println!("{}", y.sum());

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

    let b1 = weight::b1::b1.to_vec();
    let b2 = weight::b2::b2.to_vec();
    let b3 = weight::b3::b3.to_vec();

    let network = Network {
        w1: w1.clone(),
        w2: w2.clone(),
        w3: w3.clone(),
        b1: b1.clone(),
        b2: b2.clone(),
        b3: b3.clone(),
    };
    let   accuracy_cnt = 0;
    let  batch_size = 100;

    let x_test = Mnist::new().x_test;
    let y_test = Mnist::new().y_test;

    // 2D     // for i  in 0..10000{
    //     let y= Network::predict(network.clone(),  x_test.index_axis(Axis(0), i).to_owned().iter().map(|x|*x as f64).collect());
    //     let p = y.argmax().unwrap();//확률이 가장 높은 원소의 인덱스를 얻는다

    //     if p ==y_test[i] as usize{
    //          accuracy_cnt+=1
    //     }

    // }
    // println!("Accuracy:{}",accuracy_cnt as f64/10000 as f64);
    //     let start_time = Instant::now();

    //     for i in (0..10000).step_by(batch_size) {
    //         let x_batch: Array2<f64> = x_test.slice(s![i..i + batch_size, ..]).to_owned();
    //         let y_batch = Network::predict(network.clone(), &x_batch);
    //         let p: Array1<usize> = y_batch.map_axis(Axis(1), |view| view.argmax().unwrap());
    //         println!("{}",p);
    //         accuracy_cnt += y_test
    //         .slice(s![i..i + batch_size])
    //         .iter()
    //         .zip(p.iter())
    //         .filter(|&(expected, predicted)| *expected == *predicted as i64)
    //         .count();
    //     }
    //     let end_time = Instant::now();
    // let elapsed_time = end_time - start_time;
    // println!("Elapsed Time: {:?}", elapsed_time);
    println!("Accuracy:{}", accuracy_cnt as f64 / 10000 as f64);
    let start_time = Instant::now();

    let accuracy_cnt: usize = (0..10000)
        .into_par_iter()
        .step_by(batch_size)
        .map(|i| {
            let x_batch: Array2<f64> = x_test.slice(s![i..i + batch_size, ..]).to_owned();
            let y_batch = Network::predict(network.clone(), &x_batch);
            let p: Array1<usize> = y_batch.map_axis(Axis(1), |view| view.argmax().unwrap());
            y_test
                .slice(s![i..i + batch_size])
                .iter()
                .zip(p.iter())
                .filter(|&(expected, predicted)| *expected == *predicted as i64)
                .count()
        })
        .sum();
    let end_time = Instant::now();
    let elapsed_time = end_time - start_time;
    println!("Elapsed Time: {:?}", elapsed_time);

    println!("Accuracy: {}", accuracy_cnt as f64 / 10000.0);

    let a = arr1(&[0.3, 2.9, 4.0]);
    let y = softmax(&a);
    println!("{}", y);
}
//시그모이드
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/*ReLU */

fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|element| if element > 0.0 { element } else { 0.0 })
}

/*항동함수 */

fn idenity_function(x: &Array1<f64>) -> Array1<f64> {
    return x.clone();
}
#[derive(Clone)]
struct Network {
    w1: Array2<f64>,
    w2: Array2<f64>,
    w3: Array2<f64>,
    b1: Vec<f64>,
    b2: Vec<f64>,
    b3: Vec<f64>,
}
impl Network {
    // pub fn predict(self, x: Array1<f64>) -> Array1<f64> {
    //     println!("{}", 1);
    //     let (w1, w2, w3) = (self.w1, self.w2, self.w3);
    //     println!("{}", 2);

    //     let (b1, b2, b3) = (arr1(&self.b1), arr1(&self.b2), arr1(&self.b3));
    //     println!("{}", 3);
    //     println!("{:?}", x.shape());
    //     println!("{:?}", w1.shape());
    //     println!("{:?}", b1.shape());

    //     let a1 = x.dot(&w1) + &b1;
    //     println!("{}", 4);

    //     let z1 = sigmoid(&a1);
    //     let a2 = z1.dot(&w2) + &b2;
    //     let z2 = sigmoid(&a2);
    //     let a3 = z2.dot(&w3) + &b3;
    //     let y = softmax(&a3);
    //     println!("{}",y);
    //     y
    // }
    fn predict(self, x: &Array<f64, ndarray::Dim<[usize; 2]>>) -> Array2<f64> {
        let (w1, w2, w3) = (self.w1, self.w2, self.w3);
        let (b1, b2, b3) = (arr1(&self.b1), arr1(&self.b2), arr1(&self.b3));

        let a1 = x.dot(&w1) + b1;
        let z1 = a1.map(|&x| sigmoid(x));
        let a2 = z1.dot(&w2) + b2;
        let z2 = a2.map(|&x| sigmoid(x));
        let a3 = z2.dot(&w3) + b3;

        let vec_of_vec: Vec<Vec<f64>> = a3
            .axis_iter(Axis(0))
            .map(|row| softmax(&row.to_owned()))
            .collect::<Vec<_>>()
            .iter()
            .map(|x| x.to_vec())
            .collect();
        let array_2d: Array2<f64> = Array2::from_shape_vec(
            (a3.shape()[0], a3.shape()[1]),
            vec_of_vec.into_iter().flatten().collect(),
        )
        .unwrap();
        array_2d

        // let a2: Array2<f64> = arr2(&softmax_matrix.,into_iter().map(|arr| arr.to_vec()).collect::<Vec<f64>>());
    }
}
/*===========activation function========= */

fn step(x: i32) -> i32 {
    if x > 0 {
        return 1;
    } else {
        return 0;
    }
}
fn step_function(x: &Array1<f64>) -> Array1<f64> {
    x.map(|&val| if val > 0.0 { 1.0 } else { 0.0 })
}
fn softmax(a: &Array1<f64>) -> Array1<f64> {
    let c: f64 = a[a.argmax().unwrap()];
    let exp_a = a.mapv(|x| (x - c).exp()); // Subtract the maximum value and exponentiate each element
    let sum_exp_a = exp_a.sum(); // Compute the sum of the exponentiated values
    exp_a / sum_exp_a
}
struct Mnist {
    x_train: Array2<i64>,
    y_train: Array1<i64>,
    x_test: Array2<f64>,
    y_test: Array1<i64>,
}

impl Mnist {
    fn new() -> Self {
        let train_df = CsvReader::from_path("./dataset/digit-recognizer/train.csv")
            .unwrap()
            .finish()
            .unwrap();
        let test_df = CsvReader::from_path("./dataset/mnist/x_test.csv")
            .unwrap()
            .has_header(false)
            .finish()
            .unwrap();
        let submission = CsvReader::from_path("./dataset/mnist/y_test.csv")
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
        let x_train = train_df
            .drop("label")
            .unwrap()
            .to_ndarray::<Int64Type>(IndexOrder::Fortran)
            .unwrap();

        let x_test = test_df
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();

        let y_test = arr1(
            &submission
                .column("label")
                .unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<i64>>(),
        );

        Mnist {
            x_train,
            y_train,
            x_test,
            y_test,
        }
    }
}