use ndarray::prelude::*;
pub fn main(){

}

fn identity(x:f64)->f64{
    x
}
fn identity_function(x: &Array1<f64>) -> Array1<f64> {
  x.clone()
}