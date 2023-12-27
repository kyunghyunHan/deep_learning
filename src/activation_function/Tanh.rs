use ndarray::prelude::*;

 fn tanh(x:f32)->f32{
    ((x.exp() - (-x).exp()) / (x.exp() + (-x).exp()))

 }
 fn tanh_function(x: &Array1<f64>) -> Array1<f64> {
    x.map(|&val|((val.exp() - (-val).exp()) / (val.exp() + (-val).exp())))

 }

pub fn main(){
    let test_values = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]); // Test values
   println!("{}", tanh_function(&test_values))
}