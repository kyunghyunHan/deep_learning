/*관련기술 */
use ndarray::prelude::*;

pub fn main() {
    let t = arr1(&[0f64, 0f64, 1f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64]);
    let y = arr1(&[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]);
    println!("{}", cross_entropy_error(y.into_dyn(), t.into_dyn()));
}
fn cross_entropy_error(y: ArrayD<f64>, t: ArrayD<f64>) -> f64 {
    let delta = 1e-7;
    match y.ndim() {
        1 => {
            let t= t.clone().into_shape((1,t.clone().len())).unwrap();
            let y= y.clone().into_shape((1,y.clone().len())).unwrap();
            
            y
            .iter()
            .zip(t.iter())
            .map(|(&y, &t)| t * (y + delta).ln())
            .sum::<f64>()}
        2 => y
            .iter()
            .zip(t.iter())
            .map(|(&y, &t)| t * (y + delta).ln())
            .sum::<f64>(),
        _ => {
            panic!("rank error")
        }
    }
}

struct SGD {
    lr: f64,
}
impl SGD {
    fn update() {}
}

struct Momentum {}

impl Momentum {
    fn update() {}
}

struct AdaGrad {}

impl AdaGrad {
    fn update() {}
}
