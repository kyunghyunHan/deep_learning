pub struct MulLayer {
    x: Option<f64>,
    y: Option<f64>,
}
impl MulLayer {
    fn forward(&mut self, x: f64, y: f64) -> f64 {
        self.x = Some(x);
        self.y = Some(y);
        let out = x * y;
        out
    }
    fn backward(self, dout: f64) -> (f64, f64) {
        let dx = dout * self.y.unwrap();
        let dy = dout * self.x.unwrap();
        (dx, dy)
    }
}


