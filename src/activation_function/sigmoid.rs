use ndarray::{prelude::*};


fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.map(|val: &f64| 1.0 / (1.0 + f64::exp(-val)))
  }

fn sigmoid_function(x:f64)->f64{
    1.0/x+f64::exp(-x)
}
pub  fn main(){
    let x= 2;
    println!("{}",sigmoid_function(x as f64));
}