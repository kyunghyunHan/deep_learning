use ndarray::prelude::*;
use super::utils::{
    activation::*,
    random::{fill_with_random,random_choice},
    mnist::load_mnist,
    gradient_descent::{numerical_gradient,gradient_descent},
    error::{cross_entropy_error,sum_squares_error}
};
use ndarray_stats::QuantileExt;

#[derive(Clone)]
struct TwoLayerNet {
    w1: Array2<f64>,
    w2: Array2<f64>,
    b1: Array1<f64>,
    b2: Array1<f64>,
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
                ),
            w2: weight_init_std
                * fill_with_random(
                    &mut Array2::<f64>::zeros((hidden_size, output_size)),
                    &mut rng,
                ),

            b1: Array1::<f64>::zeros(hidden_size),

            b2: Array1::<f64>::zeros(output_size),
        }
    }
    fn predict(self, x: ArrayD<f64>) -> ArrayD<f64> {

        println!("predict1{}",x);
        let (w1, w2) = (self.w1, self.w2);
        let (b1, b2) = (self.b1, self.b2);
        println!("{}",w1);
        println!("{}",x.ndim());
        match x.ndim() {
            1 => {
                let x = x.into_dimensionality::<Ix1>().unwrap();
                let a1 = x.dot(&w1) + b1;
                println!("여기1");
                let z1 = sigmoid(a1.into_dyn()).into_dimensionality::<Ix1>().unwrap();
                let a2 = z1.dot(&w2) + b2;
                softmax(a2.into_dyn())
            }
            2 => {
                let x = x.into_dimensionality::<Ix2>().unwrap();

                println!("{}",x);

                let a1 = x.dot(&w1) + b1;

                println!("a1:{}",x);

                let z1 = sigmoid(a1.into_dyn()).into_dimensionality::<Ix2>().unwrap();
                println!("z1{}",z1);

                println!("여기3");
                println!("{},{}",z1.ndim(),w2.ndim());
                println!("{}",z1);
                let a2 = z1.dot(&w2) + b2;
                println!("a2{}",a2);

                println!("여기4");
                softmax(a2.into_dyn())
            }
            _ => {
                panic!("predict rank error");
            }
        }
    }

    pub fn loss(self, x: ArrayD<f64>, t: ArrayD<f64>) -> f64 {
        cross_entropy_error(x.into_dyn(), t.into_dyn())
    }
    fn accuracy(self, x: ArrayD<f64>, t: ArrayD<f64>) -> f64 {
        let rank = x.ndim();
       println!("{},{}",x.ndim(),t.ndim());
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
                println!("{}",x);
                let y = self.predict(x.clone().into_dyn());
                println!("{}",y);
                println!("여기3");

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
    /*가중치 매개변수의 기울기를 구하기 */
    fn numerical_gradient(
        self,
        _x: ArrayD<f64>,
        t: ArrayD<f64>,
    ) -> (ArrayD<f64>, ArrayD<f64>, ArrayD<f64>, ArrayD<f64>) {
        let (w1, w2, b1, b2) = (
            numerical_gradient(
                |_x| self.clone().loss(_x.clone(), t.clone()),
                self.clone().w1.into_dyn(),
            ),
            numerical_gradient(
                |_x| self.clone().loss(_x.clone(), t.clone()),
                self.clone().w2.into_dyn(),
            ),
            numerical_gradient(
                |_x| self.clone().loss(_x.clone(), t.clone()),
                self.clone().b1.into_dyn(),
            ),
            numerical_gradient(
                |_x| self.clone().loss(_x.clone(), t.clone()),
                self.clone().b2.into_dyn(),
            ),
        );
        (w1, w2, b1, b2)
    }
    /* numerical_gradient의 성능 개선판*/
    fn gradient(self, x: ArrayD<f64>, t: ArrayD<f64>) {}
}
pub fn main(){
    let mnist= load_mnist();
    let x_train = mnist.x_train;
    let y_train = mnist.y_train;
    let x_test = mnist.x_test;
    let y_test =mnist.y_test;

  //하이퍼 파라미터
  let iter_num = 10000; //반복횟수
  let train_size = x_train.clone().shape()[0];
  let batch_size = 100; //미니배치 사이즈
  let learning_rate = 0.1;
  let mut network = TwoLayerNet::new(784, 50, 10, 0.01);
  let mut train_loss_list: Vec<f64> = vec![];
  let mut train_acc_list: Vec<f64> = vec![];
  let mut test_acc_list: Vec<f64> = vec![];
  let  iter_per_epoch = usize::max(train_size / batch_size, 1);



  let y= arr1(&[0.0,1.0,2.0,3.0,4.0]);
  let t=arr1(&[2.0,7.0,0.0,9.0,4.0]);
  println!("{}",cross_entropy_error(y.into_dyn(), t.into_dyn()));
  for i in 0..iter_num {
      //미니배치 획득
      let batch_mask = random_choice(train_size, batch_size);
      let x_batch = x_train.clone().select(Axis(0), &batch_mask);
      let t_batch = y_train.clone().select(Axis(0), &batch_mask);

      //기울기 계산
      let (w1, w2, b1, b2) = network
          .clone()
          .numerical_gradient(x_batch.clone().into_dyn(), t_batch.clone().into_dyn());

      network.w1 -= &(w1.into_dimensionality::<Ix2>().unwrap() * learning_rate);
      network.w2 -= &(w2.into_dimensionality::<Ix2>().unwrap() * learning_rate);
      network.b1 -= &(b1.into_dimensionality::<Ix1>().unwrap() * learning_rate);
      network.b2 -= &(b2.into_dimensionality::<Ix1>().unwrap() * learning_rate);

      //매개변수 갱신
      println!("{}","매개변수");
      let loss = network
          .clone()
          .loss(x_batch.clone().into_dyn(), t_batch.clone().into_dyn());
      train_loss_list.push(loss);
      //학습 경과기록
      //1 epoch당 정확도 계싼
      println!("{}","애포크시작");
      // if i % iter_per_epoch == 0 {
      //     println!("{},{}",x_train.ndim(),y_train.ndim());
      //     let train_acc = network
      //         .clone()
      //         .accuracy(x_train.clone().into_dyn(), y_train.clone().into_dyn());
        
      //     let test_acc = network
      //         .clone()
      //         .accuracy(x_test.clone().into_dyn(), y_test.clone().into_dyn());
      //     train_acc_list.push(train_acc);
      //     test_acc_list.push(test_acc);
      //     println!("train_acc:{},test_acc:{}", train_acc, test_acc);
      // }
      println!("{:?}",train_loss_list);
  }
}