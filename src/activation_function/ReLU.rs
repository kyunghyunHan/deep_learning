
use ndarray::prelude::*;


pub fn main(){
    
}

fn relu_function(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|element| if element > 0.0 { element } else { 0.0 })
  }