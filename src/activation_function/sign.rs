use ndarray::prelude::*;

pub fn main(){
    
}
fn sign(x:f64)->f64{
    if x<0.0{
        return -1.0;
    }else if x==0.0{
        return 0.0;
    }else {
        return 1.0;
    }
    
}

fn sign_function(x: &Array1<f64>) -> Array1<f64> {
    x.map(|&val|if val<0.0{
        return -1.0;
    }else if val==0.0{
        return 0.0;
    }else {
        return 1.0;
    })
}