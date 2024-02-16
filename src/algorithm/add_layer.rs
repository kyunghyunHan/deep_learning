struct AddLayer {
    x: Option<f64>,
    y: Option<f64>,
}
impl AddLayer {
    fn forword(self, x: f64, y: f64) -> f64 {
        let out = x + y;
        out
    }
    fn backward(self, dout: f64) -> (f64, f64) {
        let dx = dout * 1f64;
        let dy = dout * 1f64;
        (dx, dy)
    }
}