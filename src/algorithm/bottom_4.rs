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
    let apple = 100f64;
    let apple_num = 2.0;
    let tax = 1.1;

    //계층들
    let mut mul_apple_layer = MulLayer { x: None, y: None };
    let mut mul_tax_layer = MulLayer { x: None, y: None };

    //순전파
    let apple_price = &mul_apple_layer.forward(apple, apple_num);
    let price = &mul_tax_layer.forward(*apple_price, tax);
    println!("{}", price);

    //역전파
    let dprice = 1f64;
    let (dapple_price, dtax) = &mul_tax_layer.backward(dprice);
    let (dapple, dapple_num) = &mul_apple_layer.backward(*dapple_price);
    println!("{},{},{}", dapple, dapple_num, dtax);

    /*덧셈 계층 */
   

}
