use image::math;
use ndarray::prelude::*;
use ndarray::Zip;
pub struct ReLU {
    mask: Option<ArrayD<bool>>,
}

impl ReLU {
    pub fn new() ->Self{
        Self{mask:None}
    }
    pub fn forward(&mut self,  x: Option<ArrayD<f64>>) -> Option<ArrayD<f64>> {
        if let Some(x_value) =  x {
          self.mask= Some(x_value.mapv(|x|if x<=0.0{true}else{false}));
          let out = x_value.mapv(|x|if x<=0.0{0.0}else{x}).into_dyn();
          Some(out)
        } else {
            print!("{}",1);
            None
        }
    }

    pub fn backward(&self, mut dout: Option<ArrayD<f64>>) -> Option<ArrayD<f64>> {
        if let (Some(mask), Some(mut dout)) = (&self.mask, dout) {
            Zip::from(&mut dout).and(mask).apply(|dout_val, &mask_val| {
                if mask_val {
                    *dout_val = 0.0;
                }
            });
            Some(dout)
        } else {
            None
        }
    }
}
