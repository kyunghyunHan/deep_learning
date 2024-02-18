pub struct MulLayer {
    pub x: Option<f64>,
    pub y: Option<f64>,
}
impl MulLayer {
    pub fn forward(&mut self ,x:f64,y:f64) -> f64 {
        self.x = Some(x);
        self.y= Some(y);
        let out = x * y;
        out
    }
    pub fn backward(&self, dout: f64) -> (f64, f64) {
        let dx = dout * self.y.unwrap();
        let dy = dout * self.x.unwrap();
        (dx, dy)
    }
}
