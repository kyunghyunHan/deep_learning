use ndarray::prelude::*;
use crate::algorithm::add_layer::AddLayer;
use super::utils::relu::ReLU;
use rand::prelude::*;
/*오차역전파법
계산 그래프 node,edge

순전파 =>왼쪽에서 오른쪽
역전파 =>미분을 할떄 중요!

국소적 계산을 전파

연쇄법칙

합성함수

*/
use super::mul_layer::MulLayer;

pub fn main() {
    let apple = Some(100f64);
    let apple_num = Some(2.0);
    let tax = Some(1.1);

    //계층들
    let mut mul_apple_layer = MulLayer { x: None, y: None };
    let mut mul_tax_layer = MulLayer { x: None, y: None };

    //순전파
    let apple_price = mul_apple_layer.forward(apple, apple_num);
    let price = mul_tax_layer.forward(apple_price, tax);
    println!("{}", price.unwrap());

    //역전파
    let dprice = Some(1f64);
    let (dapple_price, dtax) =  mul_tax_layer.backward(dprice);
    let (dapple, dapple_num) =  mul_apple_layer.backward(dapple_price);
    println!(
        "Mul:{},{},{}",
        dapple.unwrap(),
        dapple_num.unwrap(),
        dtax.unwrap()
    );

    /*덧셈 계층
    초기화 필요없음


    */
    let apple = Some(100f64);
    let apple_num = Some(2f64);
    let orange = Some(150f64);
    let orange_num = Some(3f64);
    let tax = Some(1.1);

    //계층들
    let mut mul_apple_layer = MulLayer::new();
    let mut mul_orange_layer = MulLayer::new();
    let mut add_apple_orange_layer: AddLayer = AddLayer::new();
    let mut mul_tax_layer = MulLayer::new();
    //순전파
    let apple_price =  mul_apple_layer.forward(apple, apple_num);
    let orange_price =  mul_orange_layer.forward(orange, orange_num);
    let all_price =  add_apple_orange_layer.forward(apple_price, orange_price);
    let price =   mul_tax_layer.forward(all_price, tax);
    //역전파 
    let dprice = Some(1f64);
    let (dall_price, dtax) =  mul_tax_layer.backward(dprice);
    let (dapple_price,dorange_price)=  add_apple_orange_layer.backward(dall_price);
    let (dorange,dorange_num) = mul_orange_layer.backward(dorange_price);
    let (dapple,dapple_num)=  mul_apple_layer.backward(dapple_price);
    println!("{:?}",mul_tax_layer);
    println!("{}",price.unwrap());
    println!("{},{},{},{},{}",dapple_num.unwrap(),dapple.unwrap(),dorange.unwrap(),dorange_num.unwrap(),dtax.unwrap());
    

    let mut relu = ReLU::new();
    let x= arr2(&[[1.0,-0.5],[-2.0,3.0]]);   
    let mask= relu.forward(Some(x.into_dyn()));
    println!("11:{}",mask.unwrap());
    let mut rng = rand::thread_rng();
    let random_numbers: Array1<f64> = Array1::from_shape_fn((2,), |_| rng.gen());
  println!("{:?}",random_numbers);
    
}
