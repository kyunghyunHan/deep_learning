pub struct AddLayer {
    pub x: Option<f64>,
    pub y: Option<f64>,
}

impl AddLayer {
    pub fn new()->Self{

        Self{
          x:Some(0.0),
          y:Some(0.0)
        }
      }
  
    pub fn forward(&mut self, x: Option<f64>, y: Option<f64>) -> Option<f64> {
        if let (Some(x_value), Some(y_value)) = (x, y) {
            self.x = Some(x_value);
            self.y = Some(y_value);
            let out = x_value + y_value;
            Some(out)
        } else {
            None
        }
    }
    pub fn backward(&mut self, dout: Option<f64>) -> (Option<f64>, Option<f64>) {
        if let Some(dout_value) = dout {
            let dx = dout_value * 1f64;
            let dy = dout_value * 1f64;
            (Some(dx), Some(dy))
        } else {
            (None, None)
        }
    }
}
