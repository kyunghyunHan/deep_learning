#[derive(Debug)]
pub struct MulLayer {
    pub x: Option<f64>,
    pub y: Option<f64>,
}
impl MulLayer {
    pub fn new()->Self{

        Self{
            x:None,
            y:None
          }
    }

    pub fn forward(&mut self, x: Option<f64>, y: Option<f64>) -> Option<f64> {
        if let (Some(x_value), Some(y_value)) = (x, y) {
            self.x = Some(x_value);
            self.y = Some(y_value);
            Some(self.x.unwrap() * self.y.unwrap())
        } else {
            None
        }
    }
    pub fn backward(&mut self, dout: Option<f64>) -> (Option<f64>,Option<f64>) {
        if let Some(dout_value) = dout {
            let dx: f64 = dout_value * self.y.unwrap();
            let dy = dout_value * self.x.unwrap();
            (Some(dx),Some(dy))
        }else{
            (None,None)
        }
     
    }
}
