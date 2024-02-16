/*오차역전파법 */
use ndarray::prelude::*;

struct ReLU {
    mask: Option<f64>,
}

impl ReLU {
    fn forward() {}
    fn backward() {}
}

struct Sigmoid {
    out: Option<f64>,
}
impl Sigmoid {
    fn forward() {}
    fn backward() {}
}

struct Affine {}

impl Affine {
    fn forward() {}
    fn backward() {}
}

struct SoftmaxWithLoss {}
impl SoftmaxWithLoss {
    fn forward() {}
    fn backward() {}
}

// pub fn main() {
//     let apple = 100.0;
//     let apple_num = 2.0;
//     let tax = 1.1;
//     let mut mul_apple_layer = MulLayer { x: None, y: None };
//     let mut mul_tax_layer = MulLayer { x: None, y: None };

//     let apple_price = mul_apple_layer.forwoad(apple, apple_num);
//     let price = mul_tax_layer.forwoad(apple_price, tax);

//     println!("{}", price);

//     //역전파
//     let dprice = 1f64;
//     let (dapple_price, dtax) = mul_tax_layer.backward(dprice);
//     let (dapple, dapple_num) = mul_apple_layer.backward(dapple_price);

//     println!("{} {} {}", dapple, dapple_num, dtax);
// }
