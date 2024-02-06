/*오차역전파법 */


struct MulLayer {
    x: Option<f64>,
    y: Option<f64>,
}
impl MulLayer {

    fn forwoad(&mut self, x: f64, y: f64) -> f64 {
        self.x = Some(x);
        self.y = Some(y);
        let out = x * y;
        out
    }
    fn backward(self, dout: f64) -> (f64, f64) {
        let dx = dout * self.y.unwrap();
        let dy = dout * self.x.unwrap();
        (dx,dy)
    }
}
pub fn main() {
    let apple= 100.0;
    let apple_num = 2.0;
    let tax= 1.1;
    let mut mul_apple_layer= MulLayer{
        x:None,
        y:None
    };
    let mut mul_tax_layer= MulLayer{
        x:None,
        y:None
    };

    let apple_price = mul_apple_layer.forwoad(apple, apple_num);
    let price = mul_tax_layer.forwoad(apple_price, tax);

    println!("{}",price);

}