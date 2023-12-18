use plotters::prelude::*;
use ndarray::prelude::*;



fn step_function(x: &Array1<f64>) -> Array1<f64> {
    x.map(|&val| if val > 0.0{ 1.0 } else { 0.0 })
}

fn prepare_data(){

}

struct Perceptron{

}
impl Perceptron {
    fn new(){
        //가중치 w와 바이어스 b룰 He normal방식으로 초기화
    }
}