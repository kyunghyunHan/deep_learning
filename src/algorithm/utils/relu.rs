use image::math;
use ndarray::prelude::*;

pub struct ReLU{
    mask:Option<ArrayD<f64>>
}

impl ReLU{
    pub fn forward(&mut self, x: Option<ArrayD<f64>>) -> Option<ArrayD<f64>> {
        match x {
            Some(x_values) => {
                let result=x_values.mapv(|x|if x <= 0.0 {0f64}else{x}).into_dyn();
                Some(result)
            },
            None => None, 
        }
    }
    pub fn backward (&mut self ,dout: Option<ArrayD<bool>>){ 
       
    }
}

