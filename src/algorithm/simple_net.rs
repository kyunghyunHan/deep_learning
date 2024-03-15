use ndarray::prelude::*;
use rand::prelude::*;
use ndarray_stats::QuantileExt;

use super::utils::{
    activation::*,
    error::{cross_entropy_error}
};
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
        let z = self.predict(x);
        let y = softmax(&z);
        let loss = cross_entropy_error(&mut y.into_dyn(), &mut t.into_dyn());
        loss
    }
}


pub fn main(){
     /*신경망 에서의 기울기 */
     let simple = SimpleNet {
        w: arr2(&[[0.0]]).into_dyn(),
    };
    let net = SimpleNet::_init_(simple);
    println!("가중치 매개변수:{:?}", net);
    let x = arr1(&[0.6, 0.9]);
    let p = net.clone().predict(x.clone().into_dyn());
    println!("predict{:?}", p);
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