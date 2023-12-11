use ndarray::prelude::*;
/*Step function */
fn step (x:f64)->f64 {
    if x>0.0{
        return 1.0
    }else{
        return 0.0
    }
}

fn step_function(x: &Array1<f64>) -> Array1<f64> {
    x.map(|&val| if val > 0.0{ 1.0 } else { 0.0 })
}

pub fn main(){
    
}